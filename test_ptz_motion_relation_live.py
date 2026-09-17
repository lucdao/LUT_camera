#!/usr/bin/env python3
"""Live, reversible test for a learned PTZ pixel-error motion relation.

The test uses one offline center-goal sample, moves to its source pose, finds a
classical SIFT keypoint near the generated source pixel, applies the predicted
absolute PTZ destination, and checks where the same keypoint lands. The camera
is restored to its starting pose in a finally block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import (  # noqa: E402
    connect_onvif,
    download_snapshot,
    read_actual_position,
    set_absolute_position,
    wait_until_position,
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def predict_delta(model: dict[str, Any], error_u: float, error_v: float, width: int, height: int) -> np.ndarray:
    ex = error_u / max(width / 2.0, 1.0)
    ey = error_v / max(height / 2.0, 1.0)
    features = np.array([ex, ey, ex * ex, ex * ey, ey * ey], dtype=np.float64)
    return features @ np.asarray(model["coefficients"], dtype=np.float64)


def sift_point_near(image: np.ndarray, expected: np.ndarray, radius: float) -> tuple[int, cv2.KeyPoint, np.ndarray]:
    sift = cv2.SIFT_create(nfeatures=3000)
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    if descriptors is None or not keypoints:
        raise RuntimeError("Không phát hiện được SIFT keypoint ở ảnh nguồn")
    distances = np.array([np.linalg.norm(np.asarray(kp.pt) - expected) for kp in keypoints])
    candidate_indices = np.flatnonzero(distances <= radius)
    if len(candidate_indices) == 0:
        raise RuntimeError(f"Không có SIFT keypoint trong bán kính {radius}px quanh {expected.tolist()}")
    selected = int(candidate_indices[np.argmax([keypoints[i].response for i in candidate_indices])])
    return selected, keypoints[selected], descriptors[selected:selected + 1]


def match_point_between_images(
    source: np.ndarray,
    destination: np.ndarray,
    expected: np.ndarray,
    radius: float,
) -> tuple[cv2.KeyPoint, cv2.KeyPoint, float]:
    sift = cv2.SIFT_create(nfeatures=3000)
    source_keypoints, source_descriptors = sift.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None)
    keypoints, descriptors = sift.detectAndCompute(cv2.cvtColor(destination, cv2.COLOR_BGR2GRAY), None)
    if source_descriptors is None or not source_keypoints:
        raise RuntimeError("Không phát hiện được SIFT keypoint ở ảnh nguồn")
    if descriptors is None or not keypoints:
        raise RuntimeError("Không phát hiện được SIFT keypoint ở ảnh đích")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(source_descriptors.astype(np.float32), descriptors.astype(np.float32), k=2)
    valid = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    candidates = [
        match for match in valid
        if np.linalg.norm(np.asarray(source_keypoints[match.queryIdx].pt) - expected) <= radius
    ]
    if not candidates:
        raise RuntimeError(f"Không có SIFT match hợp lệ trong bán kính {radius}px quanh {expected.tolist()}")
    best = min(candidates, key=lambda match: match.distance)
    return source_keypoints[best.queryIdx], keypoints[best.trainIdx], float(best.distance)


def move_and_capture(
    ptz: Any,
    profile_token: str,
    snapshot_uri: str,
    username: str,
    password: str,
    output: Path,
    target: tuple[float, float, float],
    settle_seconds: float,
    timeout_seconds: float,
    position_tolerance: float = 0.01,
) -> tuple[float, float, float]:
    request = set_absolute_position(ptz, profile_token, *target)
    ptz.AbsoluteMove(request)
    actual = wait_until_position(
        ptz,
        profile_token,
        target[0],
        target[1],
        target[2],
        settle_seconds,
        timeout_seconds,
        tolerance=position_tolerance,
    )
    download_snapshot(snapshot_uri, output, username, password, timeout_seconds)
    return actual


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--source-index", type=int, default=12)
    parser.add_argument("--destination-index", type=int, default=1)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--settle-seconds", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=25.0)
    parser.add_argument("--position-tolerance", type=float, default=0.01,
                        help="Maximum absolute pan/tilt/zoom error accepted before capture")
    parser.add_argument("--keypoint-radius", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    relation = load_json(args.relation_dir / "ptz_motion_relation_models.json")
    samples = load_json(args.relation_dir / "ptz_motion_relation_samples.json")["samples"]
    sample = next(
        item for item in samples
        if item["source_ptz_index"] == args.source_index
        and item["destination_ptz_index"] == args.destination_index
    )
    model = relation["models"]["quadratic_no_intercept"]
    width = int(relation["ptz_image_size"]["width"])
    height = int(relation["ptz_image_size"]["height"])
    delta = predict_delta(model, sample["error_to_centre_u"], sample["error_to_centre_v"], width, height)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "sample": sample,
        "model": "quadratic_no_intercept",
    }

    camera = media = profile = ptz = None
    original_position = None
    try:
        namespace = argparse.Namespace(
            host=args.host,
            port=args.port,
            user=args.user,
            password=args.password,
            onvif_timeout=args.onvif_timeout,
        )
        camera, media, profile, profile_token, snapshot_uri, ptz = connect_onvif(namespace)
        original_position = read_actual_position(ptz, profile_token, args.move_timeout)
        result["original_position"] = original_position
        source_actual = move_and_capture(
            ptz,
            profile_token,
            snapshot_uri,
            args.user,
            args.password,
            output_dir / "source.jpg",
            (sample["source_pan"], sample["source_tilt"], sample["zoom"]),
            args.settle_seconds,
            args.move_timeout,
            args.position_tolerance,
        )
        source_image = cv2.imread(str(output_dir / "source.jpg"), cv2.IMREAD_COLOR)
        if source_image is None:
            raise RuntimeError("Không đọc được ảnh nguồn live vừa chụp")
        expected = np.array([sample["source_pixel_u"], sample["source_pixel_v"]], dtype=np.float64)
        destination_path = output_dir / "destination.jpg"
        _, source_keypoint, _ = sift_point_near(source_image, expected, args.keypoint_radius)
        source_error = np.array([source_image.shape[1] / 2.0 - source_keypoint.pt[0], source_image.shape[0] / 2.0 - source_keypoint.pt[1]])
        delta = predict_delta(model, float(source_error[0]), float(source_error[1]), source_image.shape[1], source_image.shape[0])
        predicted_destination = (
            source_actual[0] + float(delta[0]),
            source_actual[1] + float(delta[1]),
            source_actual[2],
        )
        result.update({
            "selected_source_error_to_centre_px": source_error.tolist(),
            "predicted_delta_pan": float(delta[0]),
            "predicted_delta_tilt": float(delta[1]),
            "predicted_destination": predicted_destination,
        })
        destination_actual = move_and_capture(
            ptz,
            profile_token,
            snapshot_uri,
            args.user,
            args.password,
            output_dir / "destination.jpg",
            predicted_destination,
            args.settle_seconds,
            args.move_timeout,
            args.position_tolerance,
        )
        destination_image = cv2.imread(str(destination_path), cv2.IMREAD_COLOR)
        if destination_image is None:
            raise RuntimeError("Không đọc được ảnh đích live vừa chụp")
        source_keypoint, destination_keypoint, match_distance = match_point_between_images(
            source_image, destination_image, expected, args.keypoint_radius
        )
        actual_destination_error = np.array([
            destination_image.shape[1] / 2.0 - destination_keypoint.pt[0],
            destination_image.shape[0] / 2.0 - destination_keypoint.pt[1],
        ])
        result.update({
            "source_actual_position": source_actual,
            "destination_actual_position": destination_actual,
            "source_keypoint": [float(source_keypoint.pt[0]), float(source_keypoint.pt[1])],
            "destination_keypoint": [float(destination_keypoint.pt[0]), float(destination_keypoint.pt[1])],
            "match_distance": match_distance,
            "destination_error_to_centre_px": actual_destination_error.tolist(),
            "destination_error_norm_px": float(np.linalg.norm(actual_destination_error)),
            "status": "success",
        })
    finally:
        if ptz is not None and profile is not None and original_position is not None:
            try:
                restore = set_absolute_position(ptz, profile_token, *original_position)
                ptz.AbsoluteMove(restore)
                wait_until_position(
                    ptz, profile_token, *original_position,
                    args.settle_seconds, args.move_timeout,
                    tolerance=args.position_tolerance,
                )
                result["restored_position"] = original_position
            except Exception as exc:
                result["restore_error"] = repr(exc)
        (output_dir / "live_test_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
