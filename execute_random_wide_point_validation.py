#!/usr/bin/env python3
"""Execute the 30 Wide-pixel pose predictions and capture PTZ centre previews.

This script consumes the random Wide-pixel predictions produced by
``build_wide_ptz_mapping.py``. It captures one current Wide frame, marks the
30 requested pixels on it, then moves the PTZ to each predicted pan/tilt/zoom
state. Every PTZ capture has an untouched raw image and a separate preview
with a red centre marker and sample number.

Credentials come only from environment variables and are never written to
output JSON:

* ``WIDE_RTSP_URL``
* ``PTZ_PASSWORD`` (unless ``PTZ_RTSP_URL`` is supplied separately)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import (  # noqa: E402
    RTSPFrameReader,
    connect_onvif,
    read_actual_position,
    set_absolute_position,
    wait_until_position,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def redact_url(value: str | None) -> str | None:
    if not value:
        return value
    try:
        parts = urlsplit(value)
        if not parts.username:
            return value
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit(
            (parts.scheme, f"{quote(parts.username, safe='')}:***@{host}", parts.path, parts.query, parts.fragment)
        )
    except ValueError:
        return re.sub(r"(://[^:/@]+:)[^@/]+(@)", r"\1***\2", value)


def close_rtsp_reader(reader: RTSPFrameReader | None) -> None:
    """Avoid releasing OpenCV VideoCapture while its read thread is active."""

    if reader is None:
        return
    reader.stop_event.set()
    reader.thread.join(timeout=max(5.0, reader.timeout + 2.0))
    reader.capture.release()


def capture_frame(reader: RTSPFrameReader, output: Path, timeout: float) -> tuple[int, int]:
    not_before = time.monotonic()
    width, height = reader.save_newest_frame(output, not_before, timeout)
    return int(width), int(height)


def position_dict(position: Any) -> dict[str, float]:
    if isinstance(position, (list, tuple)):
        pan, tilt, zoom = position
    else:
        pan, tilt, zoom = position.pan, position.tilt, position.zoom
    return {"pan": float(pan), "tilt": float(tilt), "zoom": float(zoom)}


def write_ptz_preview(raw_path: Path, preview_path: Path, sample_index: int) -> None:
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được PTZ capture để tạo preview: {raw_path}")
    height, width = image.shape[:2]
    centre = (width // 2, height // 2)
    cv2.drawMarker(
        image,
        centre,
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(30, min(width, height) // 15),
        thickness=3,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"Wide point {sample_index:02d}",
        (40, height - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(preview_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Không ghi được preview PTZ: {preview_path}")


def mark_wide_points(image_path: Path, output_path: Path, samples: list[dict[str, Any]], source_size: tuple[int, int]) -> list[list[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được Wide image: {image_path}")
    height, width = image.shape[:2]
    source_width, source_height = source_size
    scaled_points: list[list[int]] = []
    for sample in samples:
        source_x, source_y = sample["wide_pixel"]
        x = min(width - 1, max(0, int(round(float(source_x) * width / source_width))))
        y = min(height - 1, max(0, int(round(float(source_y) * height / source_height))))
        scaled_points.append([x, y])
        cv2.circle(image, (x, y), 11, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(image, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            str(sample["sample_index"]),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"Không ghi được Wide marked image: {output_path}")
    return scaled_points


def build_ptz_rtsp_url(args: argparse.Namespace) -> str:
    if args.ptz_rtsp_url:
        return args.ptz_rtsp_url
    if not args.ptz_password:
        raise RuntimeError("Thiếu PTZ_PASSWORD hoặc --ptz-password.")
    return f"rtsp://{quote(args.ptz_user, safe='')}:{quote(args.ptz_password, safe='')}@{args.ptz_host}:554/0"


def connect_ptz(args: argparse.Namespace):
    # The repository helper uses generic host/port/user/password names.
    args.host = args.ptz_host
    args.port = args.ptz_port
    args.user = args.ptz_user
    args.password = args.ptz_password
    return connect_onvif(args)


def find_latest_prediction_file() -> Path:
    candidates = list(PIPELINE_DIR.glob("run_*/02_aliked_lightglue_wide_ptz_mapping/random_30_wide_pixel_predictions.json"))
    if not candidates:
        raise RuntimeError("Không tìm thấy random_30_wide_pixel_predictions.json.")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=None, help="Defaults to the latest random_30_wide_pixel_predictions.json.")
    parser.add_argument("--wide-url", default=os.getenv("WIDE_RTSP_URL"))
    parser.add_argument("--ptz-host", default="192.168.1.8")
    parser.add_argument("--ptz-port", type=int, default=80)
    parser.add_argument("--ptz-user", default="admin")
    parser.add_argument("--ptz-password", default=os.getenv("PTZ_PASSWORD"))
    parser.add_argument("--ptz-rtsp-url", default=os.getenv("PTZ_RTSP_URL"))
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--rtsp-timeout", type=float, default=15.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=45.0)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.wide_url:
        raise SystemExit("Thiếu WIDE_RTSP_URL hoặc --wide-url.")
    predictions_path = (args.predictions or find_latest_prediction_file()).resolve()
    data = json.loads(predictions_path.read_text(encoding="utf-8"))
    samples = sorted(data.get("samples", []), key=lambda item: int(item["sample_index"]))
    if len(samples) != 30:
        raise SystemExit(f"Cần đúng 30 sample; prediction file hiện có {len(samples)}.")
    source_size = (2560, 1440)
    mapping_path = predictions_path.parent / "wide_pixel_to_ptz_mapping.json"
    if mapping_path.is_file():
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        source_reference = mapping.get("centre_pairs", [{}])[0].get("wide_image")
    else:
        source_reference = None

    output_root = predictions_path.parent / f"03_execute_random_wide_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=False)
    wide_dir = output_root / "wide"
    raw_dir = output_root / "ptz_raw"
    preview_dir = output_root / "ptz_preview_center_marker"
    for directory in (wide_dir, raw_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = output_root / "execution_metadata.json"
    metadata: dict[str, Any] = {
        "stage": "execute_random_wide_point_predictions",
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "predictions": str(predictions_path),
        "mapping": str(mapping_path) if mapping_path.is_file() else None,
        "source_wide_reference": source_reference,
        "wide_stream_url": redact_url(args.wide_url),
        "ptz": {
            "host": args.ptz_host,
            "port": args.ptz_port,
            "user": args.ptz_user,
            "rtsp_url": redact_url(build_ptz_rtsp_url(args)),
        },
        "capture_settings": {
            "settle_seconds": args.settle,
            "move_timeout_seconds": args.move_timeout,
            "tolerance": args.tolerance,
            "stable_samples": args.stable_samples,
            "retries": args.retries,
        },
        "wide_points": [],
        "captures": [],
        "failures": [],
    }
    write_json(metadata_path, metadata)

    wide_reader: RTSPFrameReader | None = None
    ptz_reader: RTSPFrameReader | None = None
    try:
        wide_raw = wide_dir / "wide_validation_current.jpg"
        wide_reader = RTSPFrameReader(args.wide_url, args.rtsp_timeout)
        wide_width, wide_height = capture_frame(wide_reader, wide_raw, args.rtsp_timeout)
        close_rtsp_reader(wide_reader)
        wide_reader = None
        marked_wide = wide_dir / "wide_validation_30_points_marked.jpg"
        current_points = mark_wide_points(wide_raw, marked_wide, samples, source_size)
        metadata["wide_capture"] = {
            "captured_at_utc": utc_now(),
            "raw_image": str(wide_raw.relative_to(output_root)),
            "marked_image": str(marked_wide.relative_to(output_root)),
            "width": wide_width,
            "height": wide_height,
        }
        metadata["wide_points"] = [
            {
                "sample_index": int(sample["sample_index"]),
                "wide_pixel_mapping_reference": sample["wide_pixel"],
                "wide_pixel_current_capture": current_points[index],
                "predicted": {"pan": sample["pan"], "tilt": sample["tilt"], "zoom": sample["zoom"]},
            }
            for index, sample in enumerate(samples)
        ]
        write_json(metadata_path, metadata)

        rtsp_url = build_ptz_rtsp_url(args)
        camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
        metadata["ptz"].update(
            {
                "profile_name": getattr(profile, "Name", None),
                "profile_token": token,
                "snapshot_uri_available": bool(snapshot_uri),
            }
        )
        ptz_reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
        write_json(metadata_path, metadata)

        for sample in samples:
            index = int(sample["sample_index"])
            target = (float(sample["pan"]), float(sample["tilt"]), float(sample["zoom"]))
            raw_path = raw_dir / f"ptz_validation_{index:02d}.jpg"
            preview_path = preview_dir / raw_path.name
            captured = False
            for attempt in range(1, args.retries + 2):
                try:
                    request = set_absolute_position(ptz, token, *target)
                    ptz.AbsoluteMove(request)
                    actual_before = wait_until_position(
                        ptz,
                        token,
                        *target,
                        settle_seconds=args.settle,
                        timeout_seconds=args.move_timeout,
                        tolerance=args.tolerance,
                        stable_samples=args.stable_samples,
                    )
                    width, height = capture_frame(ptz_reader, raw_path, args.rtsp_timeout)
                    write_ptz_preview(raw_path, preview_path, index)
                    actual_after = read_actual_position(
                        ptz,
                        token,
                        args.onvif_timeout,
                        refresh_service=camera.create_ptz_service,
                    )
                    actual_before_dict = position_dict(actual_before)
                    actual_after_dict = position_dict(actual_after)
                    metadata["captures"].append(
                        {
                            "sample_index": index,
                            "status": "ok",
                            "attempt": attempt,
                            "captured_at_utc": utc_now(),
                            "wide_pixel": sample["wide_pixel"],
                            "predicted": {"pan": target[0], "tilt": target[1], "zoom": target[2]},
                            "actual_before_capture": actual_before_dict,
                            "actual_after_capture": actual_after_dict,
                            "actual_minus_predicted": {
                                "pan": actual_after_dict["pan"] - target[0],
                                "tilt": actual_after_dict["tilt"] - target[1],
                                "zoom": actual_after_dict["zoom"] - target[2],
                            },
                            "raw_image": str(raw_path.relative_to(output_root)),
                            "preview_image": str(preview_path.relative_to(output_root)),
                            "width": width,
                            "height": height,
                        }
                    )
                    captured = True
                    print(f"{index + 1:02d}/30 captured", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - preserve all failed position context
                    if attempt <= args.retries:
                        time.sleep(min(5.0, float(attempt)))
                        try:
                            close_rtsp_reader(ptz_reader)
                            camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
                            ptz_reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
                        except Exception:
                            pass
                    else:
                        metadata["failures"].append(
                            {
                                "sample_index": index,
                                "status": "error",
                                "attempts": attempt,
                                "wide_pixel": sample["wide_pixel"],
                                "predicted": {"pan": target[0], "tilt": target[1], "zoom": target[2]},
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(f"{index + 1:02d}/30 failed: {type(exc).__name__}", flush=True)
                        if not args.continue_on_error:
                            raise
            write_json(metadata_path, metadata)
            if not captured and not args.continue_on_error:
                break
    finally:
        close_rtsp_reader(wide_reader)
        close_rtsp_reader(ptz_reader)
        metadata["finished_at_utc"] = utc_now()
        metadata["status"] = "complete" if len(metadata["captures"]) == 30 else "incomplete"
        metadata["summary"] = {
            "requested": 30,
            "captured": len(metadata["captures"]),
            "failed": len(metadata["failures"]),
        }
        write_json(metadata_path, metadata)

    print(json.dumps({"status": metadata["status"], "output_root": str(output_root)}, ensure_ascii=False), flush=True)
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
