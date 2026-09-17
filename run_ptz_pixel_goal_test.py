#!/usr/bin/env python3
"""Click/coordinate PTZ pixel -> predict PTZ -> capture and mark the result.

The source pose is selected from the saved 30-pose table.  The pixel can be
provided with ``--u --v``; ``--interactive`` opens an OpenCV window and accepts
one left-click.  Timing starts immediately when the pixel input is received
and ends when the absolute destination PTZ coordinate has been computed.
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
from test_ptz_motion_relation_live import (
    load_json,
    match_point_between_images,
    move_and_capture,
    predict_delta,
    sift_point_near,
)


def draw_source(image: np.ndarray, clicked: np.ndarray, tracked: cv2.KeyPoint | None) -> np.ndarray:
    result = image.copy()
    centre = (image.shape[1] // 2, image.shape[0] // 2)
    cv2.drawMarker(result, centre, (0, 255, 0), cv2.MARKER_CROSS, 70, 4)
    cv2.circle(result, (int(round(clicked[0])), int(round(clicked[1]))), 18, (0, 0, 255), 4)
    cv2.putText(result, "PTZ centre", (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(result, "clicked pixel", (int(clicked[0]) + 20, int(clicked[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    if tracked is not None:
        point = (int(round(tracked.pt[0])), int(round(tracked.pt[1])))
        cv2.drawMarker(result, point, (255, 0, 0), cv2.MARKER_TILTED_CROSS, 55, 4)
        cv2.putText(result, "tracked source", (point[0] + 20, point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return result


def draw_destination(image: np.ndarray, point: cv2.KeyPoint, error: np.ndarray) -> np.ndarray:
    result = image.copy()
    centre = (image.shape[1] // 2, image.shape[0] // 2)
    target = (int(round(point.pt[0])), int(round(point.pt[1])))
    cv2.drawMarker(result, centre, (0, 255, 0), cv2.MARKER_CROSS, 70, 4)
    cv2.drawMarker(result, target, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 55, 4)
    cv2.line(result, centre, target, (255, 0, 0), 3)
    cv2.putText(result, "PTZ centre", (centre[0] + 20, centre[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(result, f"target error: {np.linalg.norm(error):.1f}px", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return result


def select_point(image: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.u is not None and args.v is not None:
        return np.array([float(args.u), float(args.v)], dtype=np.float64), "cli"
    if not args.interactive:
        raise RuntimeError("Cần truyền --u và --v hoặc dùng --interactive")
    window = "PTZ source: click one target pixel, press ESC to cancel"
    selected: list[tuple[float, float]] = []
    view = image.copy()

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            selected.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while not selected:
        cv2.imshow(window, view)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyWindow(window)
            raise RuntimeError("Đã hủy chọn pixel")
    cv2.destroyWindow(window)
    return np.array(selected[0], dtype=np.float64), "mouse"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=12)
    parser.add_argument("--u", type=float)
    parser.add_argument("--v", type=float)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=35.0)
    parser.add_argument("--position-tolerance", type=float, default=0.01,
                        help="Maximum absolute pan/tilt/zoom error accepted before capture")
    parser.add_argument("--feature-radius", type=float, default=160.0)
    parser.add_argument("--goal-tolerance-px", type=float, default=75.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.u is None) != (args.v is None):
        raise RuntimeError("Phải truyền đồng thời --u và --v")

    relation = load_json(args.relation_dir / "ptz_motion_relation_models.json")
    homographies = load_json(args.mapping_dir / "selected_homographies.json")["selected"]
    source = next(item for item in homographies if int(item["ptz_index"]) == args.source_index)
    model = relation["models"]["quadratic_no_intercept"]
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"source_index": args.source_index, "model": "quadratic_no_intercept"}

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
        clicked, input_kind = select_point(source_image, args)
        if not (0 <= clicked[0] < source_image.shape[1] and 0 <= clicked[1] < source_image.shape[0]):
            raise RuntimeError(f"Pixel ngoài ảnh: {clicked.tolist()}")

        input_received = time.perf_counter_ns()
        error = np.array([source_image.shape[1] / 2.0 - clicked[0], source_image.shape[0] / 2.0 - clicked[1]])
        delta = predict_delta(model, float(error[0]), float(error[1]), source_image.shape[1], source_image.shape[0])
        coordinate_ready = time.perf_counter_ns()
        destination_target = (source_actual[0] + float(delta[0]), source_actual[1] + float(delta[1]), source_actual[2])
        result.update({
            "input_kind": input_kind,
            "clicked_pixel": clicked.tolist(),
            "source_actual_position": source_actual,
            "source_error_to_centre_px": error.tolist(),
            "delta_pan": float(delta[0]),
            "delta_tilt": float(delta[1]),
            "predicted_destination": destination_target,
            "coordinate_compute_time_ms": (coordinate_ready - input_received) / 1_000_000.0,
        })
        try:
            _, tracked_source, _ = sift_point_near(source_image, clicked, args.feature_radius)
        except RuntimeError:
            tracked_source = None
        cv2.imwrite(str(output_dir / "source_marked.jpg"), draw_source(source_image, clicked, tracked_source))

        destination_actual = move_and_capture(
            ptz, profile_token, snapshot_uri, args.user, args.password,
            output_dir / "destination.jpg", destination_target, args.settle_seconds, args.move_timeout,
            args.position_tolerance,
        )
        destination_image = cv2.imread(str(output_dir / "destination.jpg"), cv2.IMREAD_COLOR)
        if destination_image is None:
            raise RuntimeError("Không đọc được destination.jpg")
        source_match, destination_match, match_distance = match_point_between_images(
            source_image, destination_image, clicked, args.feature_radius,
        )
        destination_error = np.array([
            destination_image.shape[1] / 2.0 - destination_match.pt[0],
            destination_image.shape[0] / 2.0 - destination_match.pt[1],
        ])
        result.update({
            "destination_actual_position": destination_actual,
            "matched_source_pixel": [float(source_match.pt[0]), float(source_match.pt[1])],
            "matched_destination_pixel": [float(destination_match.pt[0]), float(destination_match.pt[1])],
            "match_distance": match_distance,
            "destination_error_to_centre_px": destination_error.tolist(),
            "destination_error_norm_px": float(np.linalg.norm(destination_error)),
            "goal_tolerance_px": args.goal_tolerance_px,
            "goal_passed": bool(np.linalg.norm(destination_error) <= args.goal_tolerance_px),
            "movement_time_ms": None,
        })
        cv2.imwrite(str(output_dir / "destination_marked.jpg"), draw_destination(destination_image, destination_match, destination_error))
        result["status"] = "goal_passed" if result["goal_passed"] else "goal_failed"
    finally:
        if ptz is not None and profile is not None and original_position is not None:
            restore_error = None
            for _attempt in range(3):
                try:
                    restore_start = time.perf_counter_ns()
                    restore = set_absolute_position(ptz, profile_token, *original_position)
                    ptz.AbsoluteMove(restore)
                    wait_until_position(
                        ptz, profile_token, *original_position,
                        args.settle_seconds, args.move_timeout,
                        tolerance=args.position_tolerance,
                    )
                    result["restored_position"] = original_position
                    result["restore_time_ms"] = (time.perf_counter_ns() - restore_start) / 1_000_000.0
                    restore_error = None
                    break
                except Exception as exc:
                    restore_error = repr(exc)
                    time.sleep(1.0)
            if restore_error is not None:
                result["restore_error"] = restore_error
        (output_dir / "pixel_goal_test_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "goal_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
