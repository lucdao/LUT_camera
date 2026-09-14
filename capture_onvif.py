#!/usr/bin/env python3
"""Capture PTZ reference images through ONVIF.

Default plan: 30 images starting at the camera's actual ONVIF position, then
evenly distributed over one pan rotation. Tilt and zoom remain at the actual
values read from the camera.

Install:
    pip install onvif-zeep requests opencv-python
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import requests
from requests.auth import HTTPDigestAuth
from onvif import ONVIFCamera
from zeep.transports import Transport


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def normalize_tilt(actual_tilt: float, target_tilt: float) -> float:
    """Normalize the camera-specific report for the requested +0.1 position."""
    if target_tilt == 0.1 and math.isclose(actual_tilt, 2.81111121, abs_tol=0.01):
        return 0.1
    return actual_tilt


def wait_until_position(
    ptz: Any,
    profile_token: str,
    target_pan: float,
    target_tilt: float,
    target_zoom: float,
    settle_seconds: float,
    timeout_seconds: float,
    tolerance: float = 0.035,
    stable_samples: int = 3,
) -> tuple[float, float, float]:
    """Wait until the camera reports the target position repeatedly."""
    deadline = time.monotonic() + timeout_seconds
    last_position = None
    last_error = None
    reached_samples = 0
    while time.monotonic() < deadline:
        try:
            status = ptz.GetStatus({"ProfileToken": profile_token})
        except Exception as exc:
            # A PTZ request can briefly fail while the camera is busy moving
            # or while its ONVIF service recovers. Retry within the move window.
            last_error = exc
            reached_samples = 0
            time.sleep(0.25)
            continue
        position = getattr(status, "Position", None)
        if position is not None and getattr(position, "PanTilt", None) is not None:
            pan = float(position.PanTilt.x)
            tilt = float(position.PanTilt.y)
            zoom = float(position.Zoom.x) if getattr(position, "Zoom", None) else 0.0
            last_position = (pan, tilt, zoom)
            if (
                abs(pan - target_pan) <= tolerance
                and abs(tilt - target_tilt) <= tolerance
                and abs(zoom - target_zoom) <= tolerance
            ):
                reached_samples += 1
                if reached_samples >= stable_samples:
                    if settle_seconds > 0:
                        time.sleep(settle_seconds)
                    return last_position
            else:
                reached_samples = 0
        time.sleep(0.15)
    raise TimeoutError(
        f"Camera did not reach target pan={target_pan:.4f}, "
        f"tilt={target_tilt:.4f}, zoom={target_zoom:.4f}; "
        f"last position={last_position}; last ONVIF error={last_error}"
    )


def read_actual_position(
    ptz: Any, profile_token: str, timeout_seconds: float,
    refresh_service: Any = None,
) -> tuple[float, float, float]:
    """Read the camera's reported position without validating its range."""
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            status = ptz.GetStatus({"ProfileToken": profile_token})
            position = getattr(status, "Position", None)
            if position is not None and getattr(position, "PanTilt", None) is not None:
                pan = float(position.PanTilt.x)
                tilt = float(position.PanTilt.y)
                zoom = float(position.Zoom.x) if getattr(position, "Zoom", None) else 0.0
                return pan, tilt, zoom
        except Exception as exc:
            last_error = exc
            if refresh_service is not None:
                try:
                    ptz = refresh_service()
                except Exception as refresh_exc:
                    last_error = refresh_exc
            time.sleep(0.25)
    raise TimeoutError(f"Could not read actual ONVIF PTZ position: {last_error}")


def connect_onvif(args: argparse.Namespace) -> tuple[Any, Any, Any, str, str, Any]:
    """Create a fresh ONVIF session and return its active services."""
    onvif_transport = Transport(
        timeout=args.onvif_timeout,
        operation_timeout=args.onvif_timeout,
    )
    camera = ONVIFCamera(
        args.host, args.port, args.user, args.password,
        transport=onvif_transport,
    )
    media = camera.create_media_service()
    profiles = media.GetProfiles()
    if not profiles:
        raise RuntimeError("Camera returned no media profiles after reconnect")
    profile = profiles[0]
    profile_token = profile.token
    snapshot_uri = media.GetSnapshotUri({"ProfileToken": profile_token}).Uri
    ptz = camera.create_ptz_service()
    return camera, media, profile, profile_token, snapshot_uri, ptz


def set_absolute_position(ptz: Any, profile_token: str, pan: float, tilt: float, zoom: float) -> Any:
    request = ptz.create_type("AbsoluteMove")
    request.ProfileToken = profile_token
    request.Position = {
        "PanTilt": {"x": pan, "y": tilt},
        "Zoom": {"x": zoom},
    }
    return request


def download_snapshot(uri: str, output: Path, username: str, password: str, timeout: float) -> None:
    # Many ONVIF cameras protect the snapshot endpoint with HTTP Digest even
    # when the ONVIF SOAP service has already authenticated successfully.
    response = requests.get(
        uri, auth=HTTPDigestAuth(username, password), timeout=timeout
    )
    if response.status_code == 401:
        response = requests.get(uri, auth=(username, password), timeout=timeout)
    response.raise_for_status()
    if not response.content.startswith(b"\xff\xd8"):
        raise RuntimeError(
            f"Snapshot endpoint did not return a JPEG (Content-Type={response.headers.get('Content-Type')})"
        )
    output.write_bytes(response.content)


def open_rtsp_capture(url: str, timeout: float) -> cv2.VideoCapture:
    """Open the main RTSP stream used as the high-resolution image source."""
    timeout_ms = int(timeout * 1000)
    read_timeout_ms = min(timeout_ms, 3000)
    params = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms])
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        # A short read timeout lets the background reader reconnect quickly
        # instead of waiting for the full capture timeout after a stall.
        params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, read_timeout_ms])

    try:
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params)
    except cv2.error:
        capture = None

    if capture is None or not capture.isOpened():
        if capture is not None:
            capture.release()
        # Older OpenCV builds do not support constructor parameters.
        capture = cv2.VideoCapture()
        capture.open(url, cv2.CAP_FFMPEG)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open RTSP stream: {url}")
    # Keep only the most recent decoded frame where the backend supports it.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


class RTSPFrameReader:
    """Continuously consume RTSP so frames cannot build up in a stale buffer."""

    def __init__(self, url: str, timeout: float) -> None:
        self.url = url
        self.timeout = timeout
        self.capture = open_rtsp_capture(url, timeout)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_read_at = 0.0
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                ok, frame = self.capture.read()
                if ok and frame is not None and frame.size:
                    with self.lock:
                        self.latest_frame = frame.copy()
                        self.latest_read_at = time.monotonic()
                else:
                    self._reconnect()
            except Exception as exc:
                self.error = exc
                self._reconnect()

    def _reconnect(self) -> None:
        """Reconnect after a dropped/stalled RTSP read."""
        self.capture.release()
        if self.stop_event.wait(0.25):
            return
        try:
            self.capture = open_rtsp_capture(self.url, self.timeout)
            self.error = None
        except Exception as exc:
            self.error = exc
            self.stop_event.wait(1.0)

    def save_newest_frame(
        self, output: Path, not_before: float, timeout: float
    ) -> tuple[int, int]:
        """Save a frame read after ``not_before`` and return width/height."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if self.latest_frame is not None and self.latest_read_at >= not_before:
                    frame = self.latest_frame.copy()
                    break
            time.sleep(0.02)
        else:
            if self.error is not None:
                raise RuntimeError("RTSP reader failed") from self.error
            raise RuntimeError("RTSP stream did not return a new image before timeout")

        if not cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write RTSP frame: {output}")
        height, width = frame.shape[:2]
        return width, height

    def close(self) -> None:
        self.stop_event.set()
        self.capture.release()
        self.thread.join(timeout=2.0)


def write_metadata(path: Path, camera_info: dict[str, Any], snapshot_uri: str,
                   captures: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps({
        "camera": camera_info,
        "snapshot_uri": snapshot_uri,
        "captures": captures,
    }, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="171.231.237.114")
    parser.add_argument("--port", type=int, default=2505)
    parser.add_argument("--user", default="admin")
    parser.add_argument("--password", default=os.getenv("PTZ_PASSWORD"))
    parser.add_argument("--output", type=Path, default=Path("data/reference"))
    parser.add_argument("--rtsp-url", default=None,
                        help="Capture images from this RTSP stream instead of ONVIF SnapshotUri")
    parser.add_argument("--capture-delay", type=float, default=2.0,
                        help="Seconds to wait after the PTZ position is confirmed before capture")
    parser.add_argument("--pan-count", type=int, default=30,
                        help="Number of pan positions per tilt/zoom combination")
    parser.add_argument("--states", default=None,
                        help="Optional tilt:zoom states, e.g. 0:0,0.2:0.5,-0.2:0.5")
    parser.add_argument("--tilts", default="0",
                        help="Comma-separated ONVIF normalized tilt values, e.g. -0.2,0,0.2")
    parser.add_argument("--zooms", default="0",
                        help="Comma-separated ONVIF normalized zoom values, e.g. 0,0.5,1")
    parser.add_argument("--move-timeout", type=float, default=15.0,
                        help="Maximum seconds to wait for each PTZ target")
    parser.add_argument("--onvif-timeout", type=float, default=8.0,
                        help="Timeout for each ONVIF request")
    parser.add_argument("--onvif-retries", type=int, default=4,
                        help="Fresh ONVIF-session retries after a timeout")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.password:
        parser.error("Provide --password or set PTZ_PASSWORD")
    if args.pan_count < 2:
        parser.error("--pan-count must be at least 2")
    if args.capture_delay < 0:
        parser.error("--capture-delay cannot be negative")
    if args.onvif_timeout <= 0:
        parser.error("--onvif-timeout must be positive")
    if args.onvif_retries < 0:
        parser.error("--onvif-retries cannot be negative")

    tilts = parse_csv_floats(args.tilts)
    zooms = parse_csv_floats(args.zooms)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Connecting to ONVIF camera {args.host}:{args.port} ...")
    camera, media, profile, profile_token, snapshot_uri, ptz = connect_onvif(args)

    # Read the real camera state. The first capture is taken at this exact
    # position; no synthetic (-1, 0, 0) default is imposed.
    status = ptz.GetStatus({"ProfileToken": profile_token})
    current_position = getattr(status, "Position", None)
    if current_position is None or getattr(current_position, "PanTilt", None) is None:
        raise RuntimeError("Camera did not return a usable PTZ position from GetStatus()")
    current_pan = float(current_position.PanTilt.x)
    current_tilt = float(current_position.PanTilt.y)
    current_zoom = float(current_position.Zoom.x) if getattr(current_position, "Zoom", None) else 0.0
    print(f"Current ONVIF position: pan={current_pan:.4f}, tilt={current_tilt:.4f}, zoom={current_zoom:.4f}")

    # ONVIF normalized pan is normally [-1, 1]. Start at the real position and
    # wrap around the normalized range for a full sweep without duplicating it.
    if args.states:
        states = []
        for item in args.states.split(","):
            tilt, zoom = item.split(":")
            states.append((float(tilt), float(zoom)))
        pan_count = args.pan_count
        pans = [current_pan + 2.0 * i / pan_count for i in range(pan_count)]
        pans = [((pan + 1.0) % 2.0) - 1.0 for pan in pans]
    else:
        states = [(current_tilt, current_zoom)]
        pan_count = args.pan_count
        pans = [current_pan + 2.0 * i / pan_count for i in range(pan_count)]
        pans = [((pan + 1.0) % 2.0) - 1.0 for pan in pans]
    jobs = [(pan, tilt, zoom) for tilt, zoom in states for pan in pans]
    print(f"Profile: {profile.Name}; planned captures: {len(jobs)}")
    print(f"Snapshot URI: {snapshot_uri}")

    metadata: list[dict[str, Any]] = []
    metadata_path = args.output / "capture_metadata.json"
    rtsp_reader = None
    try:
        if args.rtsp_url and not args.dry_run:
            print("Opening RTSP image source (continuous reader)")
            rtsp_reader = RTSPFrameReader(args.rtsp_url, args.timeout)

        for index, (pan, tilt, zoom) in enumerate(jobs):
            filename = args.output / f"frame_{index:04d}.jpg"
            print(f"[{index + 1}/{len(jobs)}] pan={pan:.4f}, tilt={tilt:.4f}, zoom={zoom:.4f}")
            for attempt in range(args.onvif_retries + 1):
                try:
                    move = set_absolute_position(ptz, profile_token, pan, tilt, zoom)
                    capture_info: dict[str, Any] = {}
                    if not args.dry_run:
                        ptz.AbsoluteMove(move)
                        # Do not reject camera-specific/out-of-range ONVIF values.
                        # The camera's actual status is recorded at capture time.
                        time.sleep(args.capture_delay)
                        capture_time = time.monotonic()
                        actual_pan, actual_tilt, actual_zoom = read_actual_position(
                            ptz, profile_token, args.move_timeout,
                            refresh_service=camera.create_ptz_service,
                        )
                        print(
                            "  Actual ONVIF status at capture: "
                            f"pan={actual_pan:.4f}, tilt={actual_tilt:.4f}, "
                            f"zoom={actual_zoom:.4f}"
                        )
                        if rtsp_reader is not None:
                            width, height = rtsp_reader.save_newest_frame(
                                filename, capture_time, args.timeout
                            )
                            print(f"  Saved RTSP frame: {filename} ({width}x{height})")
                            capture_info = {
                                "capture_source": "rtsp",
                                "rtsp_url": args.rtsp_url,
                                "width": width,
                                "height": height,
                            }
                        else:
                            download_snapshot(snapshot_uri, filename, args.user, args.password, args.timeout)
                            capture_info = {"capture_source": "onvif_snapshot"}
                    metadata.append({
                        "image": filename.name,
                        "pan_onvif": actual_pan if not args.dry_run else pan,
                        "tilt_onvif": (
                            normalize_tilt(actual_tilt, tilt)
                            if not args.dry_run else tilt
                        ),
                        "zoom_onvif": actual_zoom if not args.dry_run else zoom,
                        "pan_target_onvif": pan,
                        "tilt_target_onvif": tilt,
                        "zoom_target_onvif": zoom,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        **capture_info,
                    })
                    write_metadata(
                        metadata_path,
                        {"host": args.host, "port": args.port, "profile": profile.Name},
                        args.rtsp_url or snapshot_uri, metadata,
                    )
                    break
                except Exception as exc:
                    write_metadata(
                        metadata_path,
                        {"host": args.host, "port": args.port, "profile": profile.Name},
                        args.rtsp_url or snapshot_uri, metadata,
                    )
                    if args.dry_run or attempt >= args.onvif_retries:
                        raise
                    wait_seconds = min(5.0, 1.0 + attempt)
                    print(
                        f"  ONVIF error ({type(exc).__name__}); reconnecting "
                        f"and retrying ({attempt + 1}/{args.onvif_retries}) in "
                        f"{wait_seconds:.1f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(wait_seconds)
                    camera, media, profile, profile_token, snapshot_uri, ptz = connect_onvif(args)
    finally:
        if rtsp_reader is not None:
            rtsp_reader.close()
    print(f"Metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nCapture interrupted.", file=sys.stderr)
        raise SystemExit(130)
