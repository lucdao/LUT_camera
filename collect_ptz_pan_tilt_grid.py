#!/usr/bin/env python3
"""Capture a resumable PTZ calibration grid from camera-reported ONVIF limits.

By default the script asks the connected camera for its AbsolutePanTilt and
AbsoluteZoom spaces, then creates 30 pan samples x 6 tilt samples (180 raw
images). No camera-specific tilt interval is hard-coded. If ONVIF returns a
non-finite, out-of-range, stale, or otherwise unusable pose, the metadata
keeps the raw report and uses the commanded target as the normalized pose.

The run can be resumed safely with ``--resume <output directory>``; completed
positions are never overwritten.
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

import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import (  # noqa: E402
    RTSPFrameReader,
    connect_onvif,
    set_absolute_position,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Danh sách không được rỗng.")
    return values


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


def redact_error(message: str, password: str | None) -> str:
    safe = message
    if password:
        safe = safe.replace(password, "***")
    return re.sub(r"(rtsp://[^:/@]+:)[^@/]+(@)", r"\1***\2", safe)


def close_rtsp_reader(reader: RTSPFrameReader | None) -> None:
    """Stop OpenCV's background reader before releasing its capture handle."""

    if reader is None:
        return
    reader.stop_event.set()
    reader.thread.join(timeout=max(5.0, reader.timeout + 2.0))
    reader.capture.release()


def capture_frame(reader: RTSPFrameReader, path: Path, timeout: float) -> tuple[int, int]:
    start = time.monotonic()
    width, height = reader.save_newest_frame(path, start, timeout)
    return int(width), int(height)


def position_dict(position: Any) -> dict[str, float]:
    if isinstance(position, (tuple, list)):
        pan, tilt, zoom = position
    else:
        pan, tilt, zoom = position.pan, position.tilt, position.zoom
    return {"pan": float(pan), "tilt": float(tilt), "zoom": float(zoom)}


def build_ptz_rtsp_url(args: argparse.Namespace) -> str:
    if args.ptz_rtsp_url:
        return args.ptz_rtsp_url
    if not args.ptz_password:
        raise RuntimeError("Thiếu PTZ_PASSWORD hoặc --ptz-password.")
    return f"rtsp://{quote(args.ptz_user, safe='')}:{quote(args.ptz_password, safe='')}@{args.ptz_host}:554/0"


def connect_ptz(args: argparse.Namespace):
    # The shared ONVIF helper expects generic connection option names.
    args.host = args.ptz_host
    args.port = args.ptz_port
    args.user = args.ptz_user
    args.password = args.ptz_password
    return connect_onvif(args)


def first_space(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def axis_range(space: Any, axis: str) -> dict[str, float]:
    axis_range_value = getattr(space, f"{axis}Range", None)
    if axis_range_value is None:
        raise RuntimeError(f"ONVIF không trả về {axis}Range cho PTZ space.")
    minimum = float(getattr(axis_range_value, "Min"))
    maximum = float(getattr(axis_range_value, "Max"))
    if not np.isfinite([minimum, maximum]).all() or minimum > maximum:
        raise RuntimeError(f"ONVIF trả về range {axis} không hợp lệ: {minimum}, {maximum}.")
    return {"min": minimum, "max": maximum}


def discover_onvif_ranges(ptz: Any, profile: Any) -> dict[str, Any]:
    """Read the active camera's coordinate spaces; do not assume camera limits."""

    ptz_configuration = getattr(profile, "PTZConfiguration", None)
    configuration_token = getattr(ptz_configuration, "token", None) if ptz_configuration is not None else None
    if not configuration_token:
        raise RuntimeError("Profile không có PTZConfiguration token để đọc giới hạn ONVIF.")
    options = ptz.GetConfigurationOptions({"ConfigurationToken": configuration_token})
    spaces = getattr(options, "Spaces", None)
    if spaces is None:
        raise RuntimeError("Camera không trả về ONVIF PTZ Spaces.")

    absolute_pan_tilt = first_space(getattr(spaces, "AbsolutePanTiltPositionSpace", None))
    if absolute_pan_tilt is None:
        raise RuntimeError("Camera không công bố AbsolutePanTiltPositionSpace.")
    absolute_zoom = first_space(getattr(spaces, "AbsoluteZoomPositionSpace", None))
    ranges: dict[str, Any] = {
        "source": "ONVIF GetConfigurationOptions / Absolute*PositionSpace",
        "configuration_token": configuration_token,
        "pan": axis_range(absolute_pan_tilt, "X"),
        "tilt": axis_range(absolute_pan_tilt, "Y"),
        "zoom": axis_range(absolute_zoom, "X") if absolute_zoom is not None else None,
    }
    return ranges


def make_axis_values(
    explicit: list[float] | None,
    count: int,
    limits: dict[str, float] | None,
    axis_name: str,
) -> list[float]:
    if explicit is not None:
        return list(explicit)
    if limits is None:
        raise RuntimeError(f"Không thể tự tạo {axis_name}: camera không công bố giới hạn ONVIF.")
    return [float(value) for value in np.linspace(limits["min"], limits["max"], count)]


def raw_position(ptz: Any, profile_token: str) -> dict[str, float]:
    status = ptz.GetStatus({"ProfileToken": profile_token})
    position = getattr(status, "Position", None)
    pan_tilt = getattr(position, "PanTilt", None) if position is not None else None
    if pan_tilt is None:
        raise RuntimeError("ONVIF GetStatus không trả về Position.PanTilt.")
    zoom_object = getattr(position, "Zoom", None)
    return {
        "pan": float(pan_tilt.x),
        "tilt": float(pan_tilt.y),
        "zoom": float(zoom_object.x) if zoom_object is not None else 0.0,
    }


def normalize_position(
    raw: dict[str, float] | None,
    target: dict[str, float],
    limits: dict[str, Any],
    tolerance: float,
    force_target_on_mismatch: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return canonical pose plus an audit record explaining every fallback."""

    normalized: dict[str, float] = {}
    invalid_axes: dict[str, str] = {}
    for axis in ("pan", "tilt", "zoom"):
        value = raw.get(axis) if raw is not None else None
        axis_limits = limits.get(axis)
        out_of_range = value is None or not np.isfinite(float(value))
        if not out_of_range and axis_limits is not None:
            out_of_range = (
                float(value) < float(axis_limits["min"]) - tolerance
                or float(value) > float(axis_limits["max"]) + tolerance
            )
        mismatch = value is not None and np.isfinite(float(value)) and abs(float(value) - target[axis]) > tolerance
        if out_of_range:
            invalid_axes[axis] = "missing_or_out_of_onvif_range"
        elif force_target_on_mismatch and mismatch:
            invalid_axes[axis] = "reported_pose_does_not_match_commanded_target"
        normalized[axis] = float(target[axis] if out_of_range or (force_target_on_mismatch and mismatch) else value)
    return normalized, {
        "mode": "commanded_target_fallback" if invalid_axes else "reported_onvif",
        "fallback_axes": invalid_axes,
    }


def wait_for_command_pose(
    ptz: Any,
    profile_token: str,
    target: dict[str, float],
    limits: dict[str, Any],
    settle_seconds: float,
    timeout_seconds: float,
    tolerance: float,
    stable_samples: int,
) -> dict[str, Any]:
    """Wait for a valid report, normalizing broken camera reports to target."""

    deadline = time.monotonic() + timeout_seconds
    last_raw: dict[str, float] | None = None
    last_error: str | None = None
    reached = 0
    unchanged_raw_samples = 0
    while time.monotonic() < deadline:
        try:
            raw = raw_position(ptz, profile_token)
            if last_raw is not None and all(abs(raw[axis] - last_raw[axis]) <= 1e-6 for axis in ("pan", "tilt", "zoom")):
                unchanged_raw_samples += 1
            else:
                unchanged_raw_samples = 0
            last_raw = raw
            normalized_for_wait, normalization = normalize_position(
                raw, target, limits, tolerance, force_target_on_mismatch=False
            )
            if all(abs(normalized_for_wait[axis] - target[axis]) <= tolerance for axis in ("pan", "tilt", "zoom")):
                reached += 1
                if reached >= stable_samples:
                    if settle_seconds > 0:
                        time.sleep(settle_seconds)
                    normalized, final_normalization = normalize_position(
                        raw, target, limits, tolerance, force_target_on_mismatch=True
                    )
                    return {
                        "raw": raw,
                        "normalized": normalized,
                        "normalization": final_normalization,
                        "wait_normalization": normalization,
                        "wait_fallback": False,
                    }
            else:
                reached = 0
                # If the camera keeps reporting the same in-range pose but it
                # never approaches the command, treat that status as stale.
                # Preserve it in *_raw and canonicalize the metadata to the
                # exact AbsoluteMove target instead of waiting the full timeout.
                if unchanged_raw_samples >= max(8, stable_samples * 4):
                    return {
                        "raw": raw,
                        "normalized": dict(target),
                        "normalization": {
                            "mode": "commanded_target_fallback",
                            "fallback_axes": {axis: "stable_but_target_mismatch" for axis in ("pan", "tilt", "zoom")},
                        },
                        "wait_normalization": normalization,
                        "wait_fallback": True,
                    }
        except Exception as exc:  # noqa: BLE001 - retry until the move window expires
            last_error = str(exc)
            reached = 0
        time.sleep(0.15)

    # A camera-specific status quirk must not poison the calibration metadata.
    # Preserve the last raw observation and mark that the canonical pose is the
    # exact command target supplied to AbsoluteMove.
    return {
        "raw": last_raw,
        "normalized": dict(target),
        "normalization": {
            "mode": "commanded_target_fallback",
            "fallback_axes": {axis: "move_timeout_or_unusable_report" for axis in ("pan", "tilt", "zoom")},
            "last_error": last_error,
        },
        "wait_normalization": None,
        "wait_fallback": True,
    }


def read_and_normalize_after_capture(
    ptz: Any,
    profile_token: str,
    target: dict[str, float],
    limits: dict[str, Any],
    timeout_seconds: float,
    tolerance: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        try:
            raw = raw_position(ptz, profile_token)
            normalized, normalization = normalize_position(
                raw, target, limits, tolerance, force_target_on_mismatch=True
            )
            return {"raw": raw, "normalized": normalized, "normalization": normalization}
        except Exception as exc:  # noqa: BLE001 - preserve fallback metadata
            last_error = str(exc)
            time.sleep(0.25)
    return {
        "raw": None,
        "normalized": dict(target),
        "normalization": {
            "mode": "commanded_target_fallback",
            "fallback_axes": {axis: "read_error" for axis in ("pan", "tilt", "zoom")},
            "last_error": last_error,
        },
    }


def build_plan(pans: list[float], tilts: list[float], zoom: float) -> list[dict[str, Any]]:
    """Use serpentine pan order to reduce unnecessary full-range rewinds."""

    plan: list[dict[str, Any]] = []
    sequence = 0
    for row, tilt in enumerate(tilts):
        columns = list(range(len(pans)))
        if row % 2:
            columns.reverse()
        for column in columns:
            plan.append(
                {
                    "sequence": sequence,
                    "row": row,
                    "column": column,
                    "requested": {"pan": pans[column], "tilt": tilt, "zoom": zoom},
                }
            )
            sequence += 1
    return plan


def load_resume_metadata(output_root: Path) -> dict[str, Any]:
    metadata_path = output_root / "capture_metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Không tìm thấy metadata để resume: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to a timestamped folder in this pipeline directory.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume an interrupted output directory.")
    parser.add_argument("--pan-count", type=int, default=30, help="Number of pan samples when --pans is omitted.")
    parser.add_argument("--tilt-count", type=int, default=6, help="Number of tilt samples when --tilts is omitted.")
    parser.add_argument("--pans", type=parse_float_list, default=None, help="Optional explicit ONVIF pan values.")
    parser.add_argument("--tilts", type=parse_float_list, default=None, help="Optional explicit ONVIF tilt values.")
    parser.add_argument("--pan-min", type=float, default=None, help="Optional pan-range override if the camera omits its ONVIF range.")
    parser.add_argument("--pan-max", type=float, default=None)
    parser.add_argument("--tilt-min", type=float, default=None, help="Optional tilt-range override if the camera omits its ONVIF range.")
    parser.add_argument("--tilt-max", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None, help="Fixed zoom; defaults to the camera-reported zoom minimum.")
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
    parser.add_argument("--continue-on-error", action="store_true", help="Complete remaining jobs after a failed position.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan: list[dict[str, Any]] = []
    if args.resume and args.output:
        raise SystemExit("Chỉ dùng một trong --output hoặc --resume.")
    if args.pan_count < 2 or args.tilt_count < 1:
        raise SystemExit("--pan-count phải >= 2 và --tilt-count phải >= 1.")
    if args.pans is not None and len(args.pans) < 2:
        raise SystemExit("--pans phải có ít nhất 2 giá trị.")
    if args.tilts is not None and len(args.tilts) < 1:
        raise SystemExit("--tilts phải có ít nhất 1 giá trị.")
    explicit_values = [*(args.pans or []), *(args.tilts or [])]
    if args.zoom is not None:
        explicit_values.append(args.zoom)
    if any(not -1.0 <= value <= 1.0 for value in explicit_values):
        raise SystemExit("Giá trị pan, tilt và zoom phải nằm trong khoảng ONVIF [-1, 1].")

    if args.resume:
        output_root = args.resume.resolve()
        metadata = load_resume_metadata(output_root)
        plan = metadata.get("plan", [])
        if not plan:
            raise SystemExit("Run resume không có plan đã lưu.")
    else:
        output_root = (
            args.output
            or PIPELINE_DIR / f"ptz_pan_tilt_grid_{args.pan_count}x{args.tilt_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        ).resolve()
        if output_root.exists():
            raise SystemExit(f"Output folder đã tồn tại: {output_root}")
        output_root.mkdir(parents=True, exist_ok=False)
        metadata = {
            "pipeline": "ptz_pan_tilt_grid",
            "status": "running",
            "started_at_utc": utc_now(),
            "finished_at_utc": None,
            "camera": {
                "onvif_host": args.ptz_host,
                "onvif_port": args.ptz_port,
                "onvif_user": args.ptz_user,
                "rtsp_url": redact_url(args.ptz_rtsp_url),
            },
            "plan": [],
            "plan_summary": None,
            "onvif_position_ranges": None,
            "capture_settings": {
                "settle_seconds": args.settle,
                "move_timeout_seconds": args.move_timeout,
                "position_tolerance": args.tolerance,
                "stable_samples": args.stable_samples,
                "retries_per_position": args.retries,
            },
            "captures": [],
            "failures": [],
        }
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "capture_metadata.json"
    completed_sequences = {int(item["sequence"]) for item in metadata.get("captures", []) if item.get("status") == "ok"}
    write_json(metadata_path, metadata)

    camera = media = profile = token = snapshot_uri = ptz = None
    reader: RTSPFrameReader | None = None
    plan = list(plan)
    try:
        rtsp_url = build_ptz_rtsp_url(args)
        camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
        onvif_ranges = discover_onvif_ranges(ptz, profile)
        metadata["onvif_position_ranges"] = onvif_ranges

        if not args.resume:
            pan_limits = onvif_ranges.get("pan")
            if args.pan_min is not None or args.pan_max is not None:
                if args.pan_min is None or args.pan_max is None:
                    raise SystemExit("Cần truyền cả --pan-min và --pan-max.")
                pan_limits = {"min": args.pan_min, "max": args.pan_max}
            tilt_limits = onvif_ranges.get("tilt")
            if args.tilt_min is not None or args.tilt_max is not None:
                if args.tilt_min is None or args.tilt_max is None:
                    raise SystemExit("Cần truyền cả --tilt-min và --tilt-max.")
                tilt_limits = {"min": args.tilt_min, "max": args.tilt_max}
            pans = make_axis_values(args.pans, args.pan_count, pan_limits, "pan")
            tilts = make_axis_values(args.tilts, args.tilt_count, tilt_limits, "tilt")
            if args.zoom is not None:
                zoom = float(args.zoom)
                zoom_source = "command_line"
            elif onvif_ranges.get("zoom") is not None:
                zoom = float(onvif_ranges["zoom"]["min"])
                zoom_source = "camera_reported_zoom_minimum"
            else:
                zoom = 0.0
                zoom_source = "fixed_zero_fallback_when_zoom_space_missing"
            plan = build_plan(pans, tilts, zoom)
            metadata["plan"] = plan
            metadata["plan_summary"] = {
                "pan_count": len(pans),
                "tilt_count": len(tilts),
                "capture_count": len(plan),
                "pan_values": pans,
                "tilt_values": tilts,
                "zoom": zoom,
                "zoom_source": zoom_source,
                "pan_order": "serpentine by tilt row",
                "range_source": "camera ONVIF Absolute*PositionSpace unless explicitly overridden",
            }
        if not plan:
            raise RuntimeError("Plan rỗng sau khi đọc giới hạn ONVIF.")
        metadata["camera"].update(
            {
                "profile_name": getattr(profile, "Name", None),
                "profile_token": token,
                "snapshot_uri_available": bool(snapshot_uri),
                "rtsp_url": redact_url(rtsp_url),
            }
        )
        reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
        write_json(metadata_path, metadata)

        pending = [job for job in plan if int(job["sequence"]) not in completed_sequences]
        print(f"Capturing {len(pending)} pending image(s) of {len(plan)} total...", flush=True)
        for ordinal, job in enumerate(pending, start=1):
            sequence = int(job["sequence"])
            target = job["requested"]
            raw_path = raw_dir / f"frame_{sequence:03d}_row{job['row']:02d}_col{job['column']:02d}.jpg"
            success = False
            for attempt in range(1, args.retries + 2):
                try:
                    request = set_absolute_position(ptz, token, float(target["pan"]), float(target["tilt"]), float(target["zoom"]))
                    ptz.AbsoluteMove(request)
                    before_state = wait_for_command_pose(
                        ptz,
                        token,
                        target,
                        onvif_ranges,
                        settle_seconds=args.settle,
                        timeout_seconds=args.move_timeout,
                        tolerance=args.tolerance,
                        stable_samples=args.stable_samples,
                    )
                    width, height = capture_frame(reader, raw_path, args.rtsp_timeout)
                    after_state = read_and_normalize_after_capture(
                        ptz,
                        token,
                        target,
                        onvif_ranges,
                        args.onvif_timeout,
                        args.tolerance,
                    )
                    metadata["captures"].append(
                        {
                            **job,
                            "status": "ok",
                            "attempt": attempt,
                            "captured_at_utc": utc_now(),
                            "raw_image": str(raw_path.relative_to(output_root)),
                            "width": width,
                            "height": height,
                            "actual_before_capture": before_state["normalized"],
                            "actual_before_capture_raw": before_state["raw"],
                            "actual_before_capture_normalization": before_state["normalization"],
                            "actual_before_capture_wait_fallback": before_state["wait_fallback"],
                            "actual_after_capture": after_state["normalized"],
                            "actual_after_capture_raw": after_state["raw"],
                            "actual_after_capture_normalization": after_state["normalization"],
                        }
                    )
                    success = True
                    print(f"{ordinal:03d}/{len(pending):03d} captured (sequence {sequence:03d})", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - persist per-pose recovery context
                    if attempt <= args.retries:
                        time.sleep(min(5.0, float(attempt)))
                        try:
                            close_rtsp_reader(reader)
                            camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
                            reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
                        except Exception:
                            pass
                    else:
                        metadata["failures"].append(
                            {
                                **job,
                                "status": "error",
                                "attempts": attempt,
                                "failed_at_utc": utc_now(),
                                "error_type": type(exc).__name__,
                                "error": redact_error(str(exc), args.ptz_password),
                            }
                        )
                        print(f"{ordinal:03d}/{len(pending):03d} failed (sequence {sequence:03d}): {type(exc).__name__}", flush=True)
                        if not args.continue_on_error:
                            raise
            write_json(metadata_path, metadata)
            if not success and not args.continue_on_error:
                break
    finally:
        close_rtsp_reader(reader)
        metadata["finished_at_utc"] = utc_now()
        metadata["status"] = "complete" if len(metadata.get("captures", [])) == len(plan) else "incomplete"
        metadata["summary"] = {
            "requested": len(plan),
            "captured": len(metadata.get("captures", [])),
            "failed": len(metadata.get("failures", [])),
        }
        write_json(metadata_path, metadata)

    print(json.dumps({"status": metadata["status"], "output_root": str(output_root)}, ensure_ascii=False), flush=True)
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
