#!/usr/bin/env python3
"""Test an out-of-calibration PTZ move using only two drawn points initially.

The source image contains the PTZ centre and the user pixel.  The destination
image contains only the PTZ centre, as requested; no SIFT or feature matching
is used anywhere in this test.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import connect_onvif, read_actual_position, set_absolute_position, wait_until_position  # noqa: E402
from test_ptz_motion_relation_live import load_json, move_and_capture, predict_delta  # noqa: E402


def draw_source(image: np.ndarray, point: np.ndarray) -> np.ndarray:
    result = image.copy()
    centre = (image.shape[1] // 2, image.shape[0] // 2)
    cv2.drawMarker(result, centre, (0, 255, 0), cv2.MARKER_CROSS, 70, 4)
    cv2.circle(result, (int(round(point[0])), int(round(point[1]))), 18, (0, 0, 255), 4)
    cv2.putText(result, "PTZ centre", (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(result, "selected pixel", (int(point[0]) + 20, int(point[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return result


def draw_destination(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    centre = (image.shape[1] // 2, image.shape[0] // 2)
    cv2.drawMarker(result, centre, (0, 255, 0), cv2.MARKER_CROSS, 70, 4)
    cv2.putText(result, "PTZ centre after move", (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=0)
    parser.add_argument("--u", type=float, required=True)
    parser.add_argument("--v", type=float, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=35.0)
    parser.add_argument("--position-tolerance", type=float, default=0.01,
                        help="Maximum absolute pan/tilt/zoom error accepted before capture")
    parser.add_argument("--goal-tolerance-px", type=float, default=75.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    relation = load_json(args.relation_dir / "ptz_motion_relation_models.json")
    homographies = load_json(args.mapping_dir / "selected_homographies.json")["selected"]
    source = next(item for item in homographies if int(item["ptz_index"]) == args.source_index)
    model = relation["models"]["quadratic_no_intercept"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"source_index": args.source_index, "model": "quadratic_no_intercept", "sift_used": False}
    ptz = profile = None
    original_position = None
    try:
        namespace = argparse.Namespace(host=args.host, port=args.port, user=args.user, password=args.password, onvif_timeout=args.onvif_timeout)
        _camera, _media, profile, profile_token, snapshot_uri, ptz = connect_onvif(namespace)
        original_position = read_actual_position(ptz, profile_token, args.move_timeout)
        result["original_position"] = original_position
        source_target = (
            float(source["actual_after_capture"]["pan"]),
            float(source["actual_after_capture"]["tilt"]),
            float(source["actual_after_capture"].get("zoom", 0.0)),
        )
        source_actual = move_and_capture(
            ptz, profile_token, snapshot_uri, args.user, args.password,
            output_dir / "source.jpg", source_target, args.settle_seconds, args.move_timeout,
            args.position_tolerance,
        )
        source_image = cv2.imread(str(output_dir / "source.jpg"), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError("Không đọc được source.jpg")
        point = np.array([args.u, args.v], dtype=np.float64)
        if not (0 <= point[0] < source_image.shape[1] and 0 <= point[1] < source_image.shape[0]):
            raise RuntimeError(f"Pixel ngoài ảnh: {point.tolist()}")
        input_received = time.perf_counter_ns()
        error = np.array([source_image.shape[1] / 2.0 - point[0], source_image.shape[0] / 2.0 - point[1]])
        delta = predict_delta(model, float(error[0]), float(error[1]), source_image.shape[1], source_image.shape[0])
        coordinate_ready = time.perf_counter_ns()
        destination_target = (source_actual[0] + float(delta[0]), source_actual[1] + float(delta[1]), source_actual[2])
        sampled_pans = [float(item["actual_after_capture"]["pan"]) for item in homographies]
        sampled_tilts = [float(item["actual_after_capture"]["tilt"]) for item in homographies]
        result.update({
            "selected_pixel": point.tolist(),
            "source_actual_position": source_actual,
            "source_error_to_centre_px": error.tolist(),
            "delta_pan": float(delta[0]),
            "delta_tilt": float(delta[1]),
            "predicted_destination": destination_target,
            "coordinate_compute_time_ms": (coordinate_ready - input_received) / 1_000_000.0,
            "sampled_pan_range": [min(sampled_pans), max(sampled_pans)],
            "sampled_tilt_range": [min(sampled_tilts), max(sampled_tilts)],
            "outside_sampled_pan_range": bool(destination_target[0] < min(sampled_pans) or destination_target[0] > max(sampled_pans)),
            "outside_sampled_tilt_range": bool(destination_target[1] < min(sampled_tilts) or destination_target[1] > max(sampled_tilts)),
        })
        cv2.imwrite(str(output_dir / "source_marked.jpg"), draw_source(source_image, point))
        destination_actual = move_and_capture(
            ptz, profile_token, snapshot_uri, args.user, args.password,
            output_dir / "destination.jpg", destination_target, args.settle_seconds, args.move_timeout,
            args.position_tolerance,
        )
        destination_image = cv2.imread(str(output_dir / "destination.jpg"), cv2.IMREAD_COLOR)
        if destination_image is None:
            raise RuntimeError("Không đọc được destination.jpg")
        cv2.imwrite(str(output_dir / "destination_centre_only.jpg"), draw_destination(destination_image))
        result.update({
            "destination_actual_position": destination_actual,
            "goal_tolerance_px": args.goal_tolerance_px,
            "status": "captured_destination_centre_only",
        })
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
                    result["restored_position"] = original_position
                    restore_error = None
                    break
                except Exception as exc:
                    restore_error = repr(exc)
                    time.sleep(1.0)
            if restore_error is not None:
                result["restore_error"] = restore_error
        (output_dir / "pixel_center_only_test_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "captured_destination_centre_only" else 1


if __name__ == "__main__":
    raise SystemExit(main())
