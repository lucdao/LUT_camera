#!/usr/bin/env python3
"""Capture a complete unknown-camera Wide/PTZ calibration run.

The output layout is intentionally compatible with ``build_wide_ptz_mapping``:

    run_.../
      00_wide_current_and_new_set/
      01_ptz_overlap_grid/

Unlike the legacy collector, PTZ pan/tilt samples are generated from the
camera's ONVIF Absolute*PositionSpace.  Raw status and the canonical command
target are both retained in metadata; a bad/stale status never contaminates
the calibration state used by the fitting stages.
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


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import RTSPFrameReader, set_absolute_position  # noqa: E402
from collect_ptz_pan_tilt_grid import (  # noqa: E402
    build_plan,
    build_ptz_rtsp_url,
    capture_frame,
    close_rtsp_reader,
    connect_ptz,
    discover_onvif_ranges,
    make_axis_values,
    position_dict,
    read_and_normalize_after_capture,
    wait_for_command_pose,
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
        return urlunsplit((parts.scheme, f"{quote(parts.username, safe='')}:***@{host}", parts.path, parts.query, parts.fragment))
    except ValueError:
        return re.sub(r"(://[^:/@]+:)[^@/]+(@)", r"\1***\2", value)


def redact_error(message: str, password: str | None) -> str:
    safe = message.replace(password, "***") if password else message
    return re.sub(r"(rtsp://[^:/@]+:)[^@/]+(@)", r"\1***\2", safe)


def image_dimensions(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Không đọc được ảnh vừa lưu: {path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def write_preview(source: Path, destination: Path) -> None:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được ảnh PTZ để tạo preview: {source}")
    height, width = image.shape[:2]
    cv2.drawMarker(
        image,
        (width // 2, height // 2),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=max(20, min(width, height) // 18),
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    if not cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Không ghi được preview: {destination}")


def capture_wide(args: argparse.Namespace, run_root: Path) -> dict[str, Any]:
    wide_root = run_root / "00_wide_current_and_new_set"
    current_dir = wide_root / "current_reference"
    new_dir = wide_root / "new_set"
    current_dir.mkdir(parents=True, exist_ok=True)
    new_dir.mkdir(parents=True, exist_ok=True)
    reader: RTSPFrameReader | None = None
    current_path = current_dir / "wide_current_reference.jpg"
    new_records: list[dict[str, Any]] = []
    try:
        reader = RTSPFrameReader(args.wide_url, args.rtsp_timeout)
        width, height = capture_frame(reader, current_path, args.rtsp_timeout)
        current_record = {
            "status": "ok",
            "captured_at_utc": utc_now(),
            "image": str(current_path.relative_to(run_root)),
            "width": width,
            "height": height,
            "source": "wide_rtsp",
        }
        for index in range(args.wide_count):
            if index:
                time.sleep(args.wide_interval)
            path = new_dir / f"wide_new_{index:04d}.jpg"
            item_width, item_height = capture_frame(reader, path, args.rtsp_timeout)
            new_records.append(
                {
                    "index": index,
                    "status": "ok",
                    "captured_at_utc": utc_now(),
                    "image": str(path.relative_to(run_root)),
                    "width": item_width,
                    "height": item_height,
                    "source": "wide_rtsp",
                }
            )
    finally:
        close_rtsp_reader(reader)
    metadata = {
        "stage": "wide_capture",
        "status": "ok",
        "captured_at_utc": utc_now(),
        "stream_url": redact_url(args.wide_url),
        "current_reference": current_record,
        "new_set": {"count_requested": args.wide_count, "count_captured": len(new_records), "images": new_records},
    }
    write_json(wide_root / "wide_capture_metadata.json", metadata)
    return {"reference_width": width, "reference_height": height, "metadata": metadata}


def capture_dynamic_grid(args: argparse.Namespace, run_root: Path, wide: dict[str, Any]) -> dict[str, Any]:
    ptz_root = run_root / "01_ptz_overlap_grid"
    raw_dir = ptz_root / "raw"
    preview_dir = ptz_root / "preview_center_marker"
    raw_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = ptz_root / "ptz_capture_metadata.json"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "pipeline": "unknown_camera_dynamic_ptz_overlap_grid",
        "stage": "ptz_overlap_capture",
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "camera": {
            "onvif_host": args.ptz_host,
            "onvif_port": args.ptz_port,
            "onvif_user": args.ptz_user,
            "rtsp_url": redact_url(args.ptz_rtsp_url),
        },
        "source_wide_reference": {
            "image": "00_wide_current_and_new_set/current_reference/wide_current_reference.jpg",
            "width": wide["reference_width"],
            "height": wide["reference_height"],
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
            "raw_images_are_unmodified": True,
            "invalid_or_stale_status_policy": "retain *_raw and normalize canonical actual_* to commanded target",
        },
        "captures": records,
        "failures": failures,
    }
    write_json(metadata_path, metadata)

    camera = media = profile = token = snapshot_uri = ptz = None
    reader: RTSPFrameReader | None = None
    plan: list[dict[str, Any]] = []
    try:
        rtsp_url = build_ptz_rtsp_url(args)
        camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
        ranges = discover_onvif_ranges(ptz, profile)
        metadata["onvif_position_ranges"] = ranges
        pan_limits = ranges.get("pan")
        tilt_limits = ranges.get("tilt")
        if args.pan_min is not None or args.pan_max is not None:
            if args.pan_min is None or args.pan_max is None:
                raise SystemExit("Cần truyền cả --pan-min và --pan-max.")
            pan_limits = {"min": args.pan_min, "max": args.pan_max}
        if args.tilt_min is not None or args.tilt_max is not None:
            if args.tilt_min is None or args.tilt_max is None:
                raise SystemExit("Cần truyền cả --tilt-min và --tilt-max.")
            tilt_limits = {"min": args.tilt_min, "max": args.tilt_max}
        pans = make_axis_values(args.pans, args.pan_count, pan_limits, "pan")
        tilts = make_axis_values(args.tilts, args.tilt_count, tilt_limits, "tilt")
        if args.zoom is not None:
            zoom = float(args.zoom)
            zoom_source = "command_line"
        elif ranges.get("zoom") is not None:
            zoom = float(ranges["zoom"]["min"])
            zoom_source = "camera_reported_zoom_minimum"
        else:
            zoom = 0.0
            zoom_source = "fixed_zero_fallback_when_zoom_space_missing"
        plan = build_plan(pans, tilts, zoom)
        # ``index`` is the stable identifier consumed by the feature-fitting stage.
        # ``wide_pixel`` is intentionally absent: before calibration the camera
        # orientation is unknown, so no false centre prior is injected.
        for job in plan:
            job["index"] = int(job["sequence"])
            job["wide_anchor_policy"] = "estimated_from_feature_mapping_after_capture"
            job["wide_anchor_fraction_nominal"] = [
                float(job["column"] / max(len(pans) - 1, 1)),
                float(job["row"] / max(len(tilts) - 1, 1)),
            ]
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
            "wide_anchor_source": "not assumed; recovered from ALIKED/LightGlue/RANSAC",
        }
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
        print(f"Capturing {len(plan)} dynamic PTZ positions...", flush=True)
        for ordinal, job in enumerate(plan, start=1):
            target = job["requested"]
            raw_path = raw_dir / f"ptz_{int(job['index']):03d}_row{int(job['row']):02d}_col{int(job['column']):02d}.jpg"
            preview_path = preview_dir / raw_path.name
            success = False
            for attempt in range(1, args.retries + 2):
                try:
                    request = set_absolute_position(ptz, token, float(target["pan"]), float(target["tilt"]), float(target["zoom"]))
                    ptz.AbsoluteMove(request)
                    before = wait_for_command_pose(
                        ptz, token, target, ranges, args.settle, args.move_timeout,
                        args.tolerance, args.stable_samples,
                    )
                    width, height = capture_frame(reader, raw_path, args.rtsp_timeout)
                    after = read_and_normalize_after_capture(
                        ptz, token, target, ranges, args.onvif_timeout, args.tolerance,
                    )
                    record = {
                        **job,
                        "status": "ok",
                        "attempt": attempt,
                        "captured_at_utc": utc_now(),
                        "raw_image": str(raw_path.relative_to(run_root)),
                        "preview_image": str(preview_path.relative_to(run_root)),
                        "width": width,
                        "height": height,
                        "requested": dict(target),
                        "actual_before_capture": before["normalized"],
                        "actual_before_capture_raw": before["raw"],
                        "actual_before_capture_normalization": before["normalization"],
                        "actual_before_capture_wait_fallback": before["wait_fallback"],
                        "actual_after_capture": after["normalized"],
                        "actual_after_capture_raw": after["raw"],
                        "actual_after_capture_normalization": after["normalization"],
                    }
                    write_preview(raw_path, preview_path)
                    records.append(record)
                    success = True
                    print(f"{ordinal:03d}/{len(plan):03d} captured", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - persist per-position recovery context
                    if attempt <= args.retries:
                        time.sleep(min(5.0, float(attempt)))
                        try:
                            close_rtsp_reader(reader)
                            camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
                            ranges = discover_onvif_ranges(ptz, profile)
                            reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
                        except Exception:
                            pass
                    else:
                        failures.append(
                            {
                                **job,
                                "status": "error",
                                "attempts": attempt,
                                "failed_at_utc": utc_now(),
                                "error_type": type(exc).__name__,
                                "error": redact_error(str(exc), args.ptz_password),
                            }
                        )
                        print(f"{ordinal:03d}/{len(plan):03d} failed: {type(exc).__name__}", flush=True)
                        if not args.continue_on_error:
                            raise
            write_json(metadata_path, metadata)
            if not success and not args.continue_on_error:
                break
    finally:
        close_rtsp_reader(reader)
        metadata["finished_at_utc"] = utc_now()
        metadata["status"] = "complete" if len(records) == len(plan) and plan else "incomplete"
        metadata["summary"] = {"requested": len(plan), "captured": len(records), "failed": len(failures)}
        write_json(metadata_path, metadata)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--wide-url", default=os.getenv("WIDE_RTSP_URL"))
    parser.add_argument("--wide-count", type=int, default=8)
    parser.add_argument("--wide-interval", type=float, default=0.5)
    parser.add_argument("--pan-count", type=int, default=30)
    parser.add_argument("--tilt-count", type=int, default=6)
    parser.add_argument("--pans", type=lambda value: [float(item) for item in value.split(",") if item.strip()])
    parser.add_argument("--tilts", type=lambda value: [float(item) for item in value.split(",") if item.strip()])
    parser.add_argument("--pan-min", type=float, default=None)
    parser.add_argument("--pan-max", type=float, default=None)
    parser.add_argument("--tilt-min", type=float, default=None)
    parser.add_argument("--tilt-max", type=float, default=None)
    parser.add_argument("--zoom", type=float, default=None)
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
    if not args.ptz_password and not args.ptz_rtsp_url:
        raise SystemExit("Thiếu PTZ_PASSWORD hoặc --ptz-rtsp-url.")
    if args.wide_count < 0 or args.pan_count < 2 or args.tilt_count < 1:
        raise SystemExit("wide-count >= 0, pan-count >= 2 và tilt-count >= 1.")
    run_root = (args.run_root or PIPELINE_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "pipeline": "unknown_camera_two_stage_polynomial",
        "version": "1.0.0",
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "run_root": str(run_root),
        "stages": [],
    }
    manifest_path = run_root / "run_metadata.json"
    write_json(manifest_path, manifest)
    return_code = 0
    try:
        wide = capture_wide(args, run_root)
        manifest["stages"].append({"name": "wide_capture", "status": "complete", "metadata": "00_wide_current_and_new_set/wide_capture_metadata.json"})
        write_json(manifest_path, manifest)
        ptz = capture_dynamic_grid(args, run_root, wide)
        manifest["stages"].append({"name": "dynamic_ptz_overlap_capture", "status": ptz["status"], "metadata": "01_ptz_overlap_grid/ptz_capture_metadata.json", "summary": ptz.get("summary")})
        manifest["status"] = ptz["status"]
        return_code = 0 if ptz["status"] == "complete" else 2
    except Exception as exc:  # noqa: BLE001 - persist failure in manifest
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = redact_error(str(exc), args.ptz_password)
        return_code = 1
    finally:
        manifest["finished_at_utc"] = utc_now()
        write_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "run_root": str(run_root)}, ensure_ascii=False), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
