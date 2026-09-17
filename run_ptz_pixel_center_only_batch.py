#!/usr/bin/env python3
"""Run a batch of PTZ centre-only pixel goal tests without SIFT."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import connect_onvif, open_rtsp_capture, read_actual_position, set_absolute_position, wait_until_position  # noqa: E402
from test_ptz_motion_relation_live import load_json, predict_delta  # noqa: E402
from run_ptz_pixel_center_only_test import draw_destination, draw_source  # noqa: E402


def move_and_capture_rtsp(
    ptz: Any,
    profile_token: str,
    rtsp_url: str,
    output: Path,
    target: tuple[float, float, float],
    settle_seconds: float,
    timeout_seconds: float,
    position_tolerance: float = 0.01,
) -> tuple[float, float, float]:
    request = set_absolute_position(ptz, profile_token, *target)
    ptz.AbsoluteMove(request)
    actual = wait_until_position(
        ptz, profile_token, *target, settle_seconds, timeout_seconds,
        tolerance=position_tolerance,
    )
    capture = open_rtsp_capture(rtsp_url, timeout_seconds)
    try:
        frame = None
        for _ in range(10):
            ok, candidate = capture.read()
            if ok and candidate is not None and candidate.size:
                frame = candidate
        if frame is None:
            raise RuntimeError(f"Không đọc được frame RTSP sau khi tới {target}")
        if not cv2.imwrite(str(output), frame):
            raise RuntimeError(f"Không ghi được ảnh {output}")
    finally:
        capture.release()
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--rtsp-url")
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=1.5)
    parser.add_argument("--move-timeout", type=float, default=35.0)
    parser.add_argument("--position-tolerance", type=float, default=0.01,
                        help="Maximum absolute pan/tilt/zoom error accepted before capture")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    relation = load_json(args.relation_dir / "ptz_motion_relation_models.json")
    homographies = load_json(args.mapping_dir / "selected_homographies.json")["selected"]
    model = relation["models"]["quadratic_no_intercept"]
    homography_by_index = {int(item["ptz_index"]): item for item in homographies}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rtsp_url = args.rtsp_url or f"rtsp://{quote(args.user, safe='')}:{quote(args.password, safe='')}@{args.host}:554/0"
    results: list[dict[str, Any]] = []
    ptz = profile = None
    original_position = None
    run_error = None
    restore_error = "not_attempted"
    try:
        namespace = argparse.Namespace(host=args.host, port=args.port, user=args.user, password=args.password, onvif_timeout=args.onvif_timeout)
        _camera, _media, profile, profile_token, _snapshot_uri, ptz = connect_onvif(namespace)
        original_position = read_actual_position(ptz, profile_token, args.move_timeout)
        for case in manifest["cases"]:
            case_dir = args.output_dir / case["case_id"]
            case_dir.mkdir(parents=True, exist_ok=True)
            source_item = homography_by_index[int(case["source_ptz_index"])]
            source_target = (
                float(source_item["actual_after_capture"]["pan"]),
                float(source_item["actual_after_capture"]["tilt"]),
                float(source_item["actual_after_capture"].get("zoom", 0.0)),
            )
            source_actual = move_and_capture_rtsp(
                ptz, profile_token, rtsp_url,
                case_dir / "source.jpg", source_target, args.settle_seconds, args.move_timeout,
                args.position_tolerance,
            )
            source_image = cv2.imread(str(case_dir / "source.jpg"), cv2.IMREAD_COLOR)
            if source_image is None:
                raise RuntimeError(f"Không đọc được ảnh nguồn {case['case_id']}")
            point = np.asarray(case["pixel"], dtype=np.float64)
            input_received = time.perf_counter_ns()
            error = np.array([source_image.shape[1] / 2.0 - point[0], source_image.shape[0] / 2.0 - point[1]])
            delta = predict_delta(model, float(error[0]), float(error[1]), source_image.shape[1], source_image.shape[0])
            coordinate_ready = time.perf_counter_ns()
            destination_target = (source_actual[0] + float(delta[0]), source_actual[1] + float(delta[1]), source_actual[2])
            cv2.imwrite(str(case_dir / "source_marked.jpg"), draw_source(source_image, point))
            destination_actual = move_and_capture_rtsp(
                ptz, profile_token, rtsp_url,
                case_dir / "destination.jpg", destination_target, args.settle_seconds, args.move_timeout,
                args.position_tolerance,
            )
            destination_image = cv2.imread(str(case_dir / "destination.jpg"), cv2.IMREAD_COLOR)
            if destination_image is None:
                raise RuntimeError(f"Không đọc được ảnh đích {case['case_id']}")
            cv2.imwrite(str(case_dir / "destination_centre_only.jpg"), draw_destination(destination_image))
            results.append({
                **case,
                "source_actual_position": source_actual,
                "source_error_to_centre_px": error.tolist(),
                "computed_delta_pan": float(delta[0]),
                "computed_delta_tilt": float(delta[1]),
                "computed_destination": destination_target,
                "actual_destination_position": destination_actual,
                "coordinate_compute_time_ms": (coordinate_ready - input_received) / 1_000_000.0,
                "sift_used": False,
                "status": "captured_destination_centre_only",
            })
    except Exception as exc:
        run_error = repr(exc)
    finally:
        if ptz is not None and profile is not None and original_position is not None:
            restore_error = None
            for _attempt in range(3):
                try:
                    restore = set_absolute_position(ptz, profile_token, *original_position)
                    ptz.AbsoluteMove(restore)
                    wait_until_position(
                        ptz, profile_token, *original_position,
                        args.settle_seconds, args.move_timeout,
                        tolerance=args.position_tolerance,
                    )
                    restore_error = None
                    break
                except Exception as exc:
                    restore_error = repr(exc)
                    time.sleep(1.0)
    report = {
        "manifest": str(args.manifest),
        "sift_used": False,
        "case_count": len(results),
        "outside_wide_count": sum(item["category"] == "outside_wide" for item in results),
        "original_position": original_position,
        "restored": restore_error is None if original_position is not None else False,
        "restore_error": restore_error if original_position is not None else "not_connected",
        "run_error": run_error,
        "results": results,
    }
    (args.output_dir / "batch_result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "case_count": len(results), "outside_wide_count": report["outside_wide_count"], "restored": report["restored"], "output_dir": str(args.output_dir)}, ensure_ascii=False))
    return 0 if len(results) == len(manifest["cases"]) and report["restored"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
