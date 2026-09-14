#!/usr/bin/env python3
"""Build a Wide-pixel -> PTZ pose mapping from an acquired Wide/PTZ run.

Workflow
========
1. Extract ALIKED local features from the Wide reference/new-set images and
   each captured PTZ image.
2. Match every Wide/PTZ pair with LightGlue, estimate ``H_ptz_to_wide`` using
   MAGSAC RANSAC, and reject geometrically implausible matches.
3. Map the PTZ image centre through each selected homography to obtain
   (Wide x, Wide y) <-> (actual pan, tilt, zoom) correspondences.
4. Fit a global projective matrix ``H_wide_to_lattice`` into the calibrated
   PTZ lattice, then use a shape-preserving 2-D PCHIP pose surface whenever
   the captured pan/tilt grid is measurably non-linear.
5. Emit 30 deterministic random Wide-pixel pose predictions. This stage is
   prediction-only: it never moves the camera.

A homography is a planar approximation. The reliability filters deliberately
use the saved acquisition anchor only as a validation prior; camera state and
the centre-to-Wide correspondence still come from the feature/RANSAC result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

try:
    import kornia.feature as KF
except ImportError as exc:  # pragma: no cover - a clear startup error is enough
    raise SystemExit("Thiếu Kornia. Cài kornia>=0.8 để dùng ALIKED + LightGlue.") from exc


PIPELINE_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    """Convert NumPy / Path values so they can be written as JSON."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def find_latest_complete_run() -> Path:
    candidates: list[Path] = []
    for directory in PIPELINE_DIR.glob("run_*"):
        manifest = directory / "run_metadata.json"
        if not manifest.is_file():
            continue
        try:
            if read_json(manifest).get("status") == "complete":
                candidates.append(directory)
        except (json.JSONDecodeError, OSError):
            continue
    if not candidates:
        raise RuntimeError("Không tìm thấy run Wide/PTZ hoàn chỉnh trong thư mục pipeline.")
    return max(candidates, key=lambda item: item.stat().st_mtime)


def load_run(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    wide_metadata_path = run_root / "00_wide_current_and_new_set" / "wide_capture_metadata.json"
    ptz_metadata_path = run_root / "01_ptz_overlap_grid" / "ptz_capture_metadata.json"
    if not wide_metadata_path.is_file() or not ptz_metadata_path.is_file():
        raise RuntimeError(f"Không phải run Wide/PTZ hợp lệ: {run_root}")

    wide_metadata = read_json(wide_metadata_path)
    ptz_metadata = read_json(ptz_metadata_path)
    if ptz_metadata.get("status") != "complete":
        raise RuntimeError("Run PTZ chưa complete; không fit ánh xạ từ tập chưa hoàn chỉnh.")

    wide_frames: list[dict[str, Any]] = []
    current = wide_metadata.get("current_reference")
    if current and current.get("status") == "ok":
        wide_frames.append({"kind": "current_reference", **current})
    for item in wide_metadata.get("new_set", {}).get("images", []):
        if item.get("status") == "ok":
            wide_frames.append({"kind": "new_set", **item})
    if not wide_frames:
        raise RuntimeError("Không tìm thấy ảnh Wide hợp lệ trong metadata.")

    captures = sorted(
        (item for item in ptz_metadata.get("captures", []) if item.get("status") == "ok"),
        key=lambda item: int(item["index"]),
    )
    if len(captures) < 4:
        raise RuntimeError("Cần tối thiểu 4 ảnh PTZ để fit homography.")
    return wide_frames, captures, ptz_metadata


@dataclass
class FeatureRecord:
    identifier: str
    image_path: Path
    width: int
    height: int
    work_width: int
    work_height: int
    keypoints_work: torch.Tensor
    keypoints_original: np.ndarray
    descriptors: torch.Tensor
    scores: np.ndarray


def load_tensor(path: Path, max_side: int, device: torch.device) -> tuple[torch.Tensor, int, int, int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không mở được ảnh: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(width, height)))
    work_width = max(32, int(round(width * scale)))
    work_height = max(32, int(round(height * scale)))
    if (work_width, work_height) != (width, height):
        image = cv2.resize(image, (work_width, work_height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    return tensor, width, height, work_width, work_height


def extract_features(
    identifier: str,
    image_path: Path,
    extractor: Any,
    max_side: int,
    device: torch.device,
    feature_dir: Path,
) -> FeatureRecord:
    tensor, width, height, work_width, work_height = load_tensor(image_path, max_side, device)
    with torch.inference_mode():
        features = extractor(tensor)[0]
    keypoints_work = features.keypoints.detach()
    keypoints_original = keypoints_work.cpu().numpy().astype(np.float32, copy=True)
    keypoints_original[:, 0] *= width / work_width
    keypoints_original[:, 1] *= height / work_height
    scores = features.keypoint_scores.detach().cpu().numpy().astype(np.float32, copy=False)
    descriptors = features.descriptors.detach()
    np.savez_compressed(
        feature_dir / f"{identifier}.npz",
        image=str(image_path),
        original_size=np.asarray([width, height], dtype=np.int32),
        working_size=np.asarray([work_width, work_height], dtype=np.int32),
        keypoints_xy=keypoints_original,
        scores=scores,
        descriptors=descriptors.cpu().numpy().astype(np.float32, copy=False),
    )
    return FeatureRecord(
        identifier=identifier,
        image_path=image_path,
        width=width,
        height=height,
        work_width=work_width,
        work_height=work_height,
        keypoints_work=keypoints_work,
        keypoints_original=keypoints_original,
        descriptors=descriptors,
        scores=scores,
    )


def point_transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def finite_points(points: np.ndarray) -> bool:
    return bool(np.isfinite(points).all())


def homography_candidate(
    *,
    wide: FeatureRecord,
    ptz: FeatureRecord,
    capture: dict[str, Any],
    matcher: Any,
    device: torch.device,
    args: argparse.Namespace,
    match_dir: Path,
    run_root: Path,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Match one Wide/PTZ pair and return its homography quality record."""

    input_data = {
        "image0": {
            "keypoints": wide.keypoints_work.unsqueeze(0),
            "descriptors": wide.descriptors.unsqueeze(0),
            "image_size": torch.tensor([[wide.work_width, wide.work_height]], device=device),
        },
        "image1": {
            "keypoints": ptz.keypoints_work.unsqueeze(0),
            "descriptors": ptz.descriptors.unsqueeze(0),
            "image_size": torch.tensor([[ptz.work_width, ptz.work_height]], device=device),
        },
    }
    with torch.inference_mode():
        matched = matcher(input_data)
    match_indexes = matched["matches"][0].detach().cpu().numpy().astype(np.int32, copy=False)
    match_scores = matched["scores"][0].detach().cpu().numpy().astype(np.float32, copy=False)
    wide_points = (
        wide.keypoints_original[match_indexes[:, 0]].astype(np.float32, copy=False)
        if len(match_indexes)
        else np.empty((0, 2), dtype=np.float32)
    )
    ptz_points = (
        ptz.keypoints_original[match_indexes[:, 1]].astype(np.float32, copy=False)
        if len(match_indexes)
        else np.empty((0, 2), dtype=np.float32)
    )

    homography: np.ndarray | None = None
    inlier_mask = np.zeros(len(match_indexes), dtype=bool)
    projected_center: np.ndarray | None = None
    projected_corners: np.ndarray | None = None
    median_error = math.inf
    area_ratio = 0.0
    if len(match_indexes) >= 4:
        homography, inliers = cv2.findHomography(
            ptz_points,
            wide_points,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=args.ransac_threshold,
            maxIters=10000,
            confidence=0.999,
        )
        if homography is not None and inliers is not None:
            homography = homography / homography[2, 2]
            inlier_mask = inliers.ravel().astype(bool)
            reprojected = point_transform(ptz_points, homography)
            errors = np.linalg.norm(reprojected - wide_points, axis=1)
            if inlier_mask.any():
                median_error = float(np.median(errors[inlier_mask]))
            center = np.array([[ptz.width / 2.0, ptz.height / 2.0]], dtype=np.float32)
            corners = np.array(
                [[0.0, 0.0], [ptz.width - 1.0, 0.0], [ptz.width - 1.0, ptz.height - 1.0], [0.0, ptz.height - 1.0]],
                dtype=np.float32,
            )
            projected_center = point_transform(center, homography)[0]
            projected_corners = point_transform(corners, homography)
            if finite_points(projected_corners):
                area_ratio = abs(float(cv2.contourArea(projected_corners.astype(np.float32)))) / float(wide.width * wide.height)

    expected_value = capture.get("wide_pixel")
    expected = np.asarray(expected_value if expected_value is not None else [math.nan, math.nan], dtype=np.float64)
    center_prior_available = bool(np.isfinite(expected).all())
    center_prior_error = (
        float(np.linalg.norm(projected_center.astype(np.float64) - expected))
        if projected_center is not None and finite_points(projected_center) and center_prior_available
        else 0.0 if not center_prior_available else math.inf
    )
    center_inside_strict = bool(
        projected_center is not None
        and finite_points(projected_center)
        and 0.0 <= projected_center[0] < wide.width
        and 0.0 <= projected_center[1] < wide.height
    )
    center_within_margin = bool(
        projected_center is not None
        and finite_points(projected_center)
        and -args.center_boundary_margin <= projected_center[0] < wide.width + args.center_boundary_margin
        and -args.center_boundary_margin <= projected_center[1] < wide.height + args.center_boundary_margin
    )
    inlier_count = int(inlier_mask.sum())
    inlier_ratio = float(inlier_count / len(match_indexes)) if len(match_indexes) else 0.0
    plausible = bool(
        homography is not None
        and center_within_margin
        and inlier_count >= args.min_inliers
        and inlier_ratio >= args.min_inlier_ratio
        and median_error <= args.max_median_reprojection_error
        and args.min_area_ratio <= area_ratio <= args.max_area_ratio
        and (not center_prior_available or center_prior_error <= args.max_center_prior_error)
    )
    score = (
        3.0 * inlier_count
        + 50.0 * inlier_ratio
        - 2.0 * (median_error if math.isfinite(median_error) else 1000.0)
        - (center_prior_error / 40.0 if center_prior_available else 0.0)
    )

    stem = f"ptz_{int(capture['index']):02d}__{wide.identifier}"
    np.savez_compressed(
        match_dir / f"{stem}.npz",
        wide_points_xy=wide_points,
        ptz_points_xy=ptz_points,
        scores=match_scores,
        inlier_mask=inlier_mask,
        H_ptz_to_wide=(homography if homography is not None else np.empty((0, 0), dtype=np.float64)),
    )
    result = {
        "ptz_index": int(capture["index"]),
        "wide_id": wide.identifier,
        "wide_image": relative(wide.image_path, run_root),
        "ptz_image": relative(ptz.image_path, run_root),
        "match_file": relative(match_dir / f"{stem}.npz", run_root),
        "matches": int(len(match_indexes)),
        "inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "median_reprojection_error_px": None if not math.isfinite(median_error) else median_error,
        "H_ptz_to_wide": None if homography is None else homography.tolist(),
        "mapped_ptz_center_on_wide": None if projected_center is None else projected_center.tolist(),
        "projected_ptz_corners_on_wide": None if projected_corners is None else projected_corners.tolist(),
        "projected_ptz_area_ratio": area_ratio,
        "expected_wide_anchor": expected.tolist() if center_prior_available else None,
        "center_prior_used": center_prior_available,
        "center_prior_error_px": None if not math.isfinite(center_prior_error) else center_prior_error,
        "center_inside_wide": center_inside_strict,
        "center_within_boundary_margin": center_within_margin,
        "plausible": plausible,
        "selection_score": score,
    }
    return result, wide_points, ptz_points, inlier_mask, homography


def save_match_visualization(
    path: Path,
    wide_path: Path,
    ptz_path: Path,
    wide_points: np.ndarray,
    ptz_points: np.ndarray,
    inlier_mask: np.ndarray,
) -> None:
    """Write a compact inlier-match visualization for a selected pair."""

    wide = cv2.imread(str(wide_path), cv2.IMREAD_COLOR)
    ptz = cv2.imread(str(ptz_path), cv2.IMREAD_COLOR)
    if wide is None or ptz is None or not len(wide_points):
        return
    target_width = 640
    wide_scale = target_width / wide.shape[1]
    ptz_scale = target_width / ptz.shape[1]
    wide_small = cv2.resize(wide, (target_width, int(round(wide.shape[0] * wide_scale))))
    ptz_small = cv2.resize(ptz, (target_width, int(round(ptz.shape[0] * ptz_scale))))
    canvas_height = max(wide_small.shape[0], ptz_small.shape[0])
    canvas = np.zeros((canvas_height, wide_small.shape[1] + ptz_small.shape[1], 3), dtype=np.uint8)
    canvas[: wide_small.shape[0], : wide_small.shape[1]] = wide_small
    canvas[: ptz_small.shape[0], wide_small.shape[1] :] = ptz_small
    indexes = np.flatnonzero(inlier_mask)[:80]
    rng = np.random.default_rng(20260908)
    for index in indexes:
        color = tuple(int(value) for value in rng.integers(50, 256, size=3))
        wide_xy = tuple(np.round(wide_points[index] * wide_scale).astype(int))
        ptz_xy = tuple((np.round(ptz_points[index] * ptz_scale) + np.array([wide_small.shape[1], 0])).astype(int))
        cv2.line(canvas, wide_xy, ptz_xy, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, wide_xy, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, ptz_xy, 2, color, -1, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def make_center_overlay(
    output_path: Path,
    wide_path: Path,
    selected: list[dict[str, Any]],
) -> None:
    image = cv2.imread(str(wide_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for item in selected:
        actual = tuple(np.round(item["mapped_ptz_center_on_wide"]).astype(int))
        expected_value = item.get("expected_wide_anchor")
        expected = tuple(np.round(expected_value).astype(int)) if expected_value is not None else None
        if expected is not None:
            cv2.drawMarker(image, expected, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 18, 2, cv2.LINE_AA)
        cv2.circle(image, actual, 7, (0, 220, 0), 2, cv2.LINE_AA)
        if expected is not None:
            cv2.line(image, expected, actual, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(item["ptz_index"]), (actual[0] + 8, actual[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def make_sample_overlay(output_path: Path, wide_path: Path, samples: list[dict[str, Any]]) -> None:
    image = cv2.imread(str(wide_path), cv2.IMREAD_COLOR)
    if image is None:
        return
    for sample in samples:
        point = tuple(int(value) for value in sample["wide_pixel"])
        cv2.circle(image, point, 7, (255, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(image, str(sample["sample_index"]), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def fit_homography_wide_to_lattice(
    selected: list[dict[str, Any]], args: argparse.Namespace
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([item["mapped_ptz_center_on_wide"] for item in selected], dtype=np.float32)
    lattice = np.asarray([[item["column"], item["row"]] for item in selected], dtype=np.float32)
    homography, mask = cv2.findHomography(
        centers,
        lattice,
        method=cv2.USAC_MAGSAC,
        ransacReprojThreshold=args.lattice_ransac_threshold,
        maxIters=10000,
        confidence=0.999,
    )
    if homography is None or mask is None:
        raise RuntimeError("Không fit được H tổng quát Wide -> PTZ lattice.")
    homography = homography / homography[2, 2]
    inliers = mask.ravel().astype(bool)
    inferred = point_transform(centers, homography)
    errors = np.linalg.norm(inferred - lattice, axis=1)
    return homography, inliers, centers, lattice, errors


def state_grid(captures: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = max(int(item["row"]) for item in captures) + 1
    columns = max(int(item["column"]) for item in captures) + 1
    grid = np.full((rows, columns, 3), np.nan, dtype=np.float64)
    for item in captures:
        position = item["actual_after_capture"]
        grid[int(item["row"]), int(item["column"])] = [
            float(position["pan"]),
            float(position["tilt"]),
            float(position["zoom"]),
        ]
    if not np.isfinite(grid).all():
        raise RuntimeError("Thiếu actual_after_capture ở ít nhất một ô PTZ grid.")
    return grid, np.arange(columns, dtype=np.float64), np.arange(rows, dtype=np.float64)


class PoseSurface:
    """Affine or separable 2-D shape-preserving PCHIP pose surface."""

    def __init__(self, grid: np.ndarray, use_pchip: bool) -> None:
        self.grid = grid
        self.rows, self.columns, _ = grid.shape
        self.u_values = np.arange(self.columns, dtype=np.float64)
        self.v_values = np.arange(self.rows, dtype=np.float64)
        self.use_pchip = use_pchip
        uu, vv = np.meshgrid(self.u_values, self.v_values)
        design = np.column_stack([np.ones(uu.size), uu.ravel(), vv.ravel()])
        self.affine_coefficients = np.linalg.lstsq(design, grid.reshape(-1, 3), rcond=None)[0]

    def predict(self, u: float, v: float) -> np.ndarray:
        if not self.use_pchip:
            return np.array([1.0, u, v], dtype=np.float64) @ self.affine_coefficients
        output = np.empty(3, dtype=np.float64)
        for axis in range(3):
            across_rows = np.array(
                [
                    float(PchipInterpolator(self.u_values, self.grid[row, :, axis], extrapolate=True)(u))
                    for row in range(self.rows)
                ],
                dtype=np.float64,
            )
            output[axis] = float(PchipInterpolator(self.v_values, across_rows, extrapolate=True)(v))
        return output


def linearity_probe(grid: np.ndarray) -> dict[str, Any]:
    rows, columns, _ = grid.shape
    uu, vv = np.meshgrid(np.arange(columns, dtype=np.float64), np.arange(rows, dtype=np.float64))
    design = np.column_stack([np.ones(uu.size), uu.ravel(), vv.ravel()])
    coefficients = np.linalg.lstsq(design, grid.reshape(-1, 3), rcond=None)[0]
    predicted = design @ coefficients
    errors = predicted - grid.reshape(-1, 3)
    rmse = np.sqrt(np.mean(errors**2, axis=0))
    return {
        "affine_coefficients_intercept_u_v": coefficients.tolist(),
        "affine_rmse": {"pan": float(rmse[0]), "tilt": float(rmse[1]), "zoom": float(rmse[2])},
    }


def pose_for_pixel(
    pixel_xy: np.ndarray,
    h_wide_to_lattice: np.ndarray,
    surface: PoseSurface,
    clamp: bool = True,
) -> dict[str, Any]:
    raw_uv = point_transform(np.asarray(pixel_xy, dtype=np.float32).reshape(1, 2), h_wide_to_lattice)[0]
    u, v = float(raw_uv[0]), float(raw_uv[1])
    clamped = False
    if clamp:
        clamped_u = min(max(u, 0.0), surface.columns - 1.0)
        clamped_v = min(max(v, 0.0), surface.rows - 1.0)
        clamped = not (math.isclose(u, clamped_u) and math.isclose(v, clamped_v))
        u, v = clamped_u, clamped_v
    pose = surface.predict(u, v)
    return {
        "lattice_uv_raw": [float(raw_uv[0]), float(raw_uv[1])],
        "lattice_uv_used": [u, v],
        "clamped_to_calibrated_lattice": clamped,
        "pan": float(pose[0]),
        "tilt": float(pose[1]),
        "zoom": float(pose[2]),
    }


def make_random_predictions(
    *,
    h_wide_to_lattice: np.ndarray,
    surface: PoseSurface,
    wide_width: int,
    wide_height: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    inverse = np.linalg.inv(h_wide_to_lattice)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    attempts = 0
    while len(samples) < count and attempts < count * 100:
        attempts += 1
        # Sampling lattice coordinates uniformly produces points distributed
        # throughout the geometrically calibrated part of the Wide image.
        u = float(rng.uniform(0.0, surface.columns - 1.0))
        v = float(rng.uniform(0.0, surface.rows - 1.0))
        wide_xy = point_transform(np.asarray([[u, v]], dtype=np.float32), inverse)[0]
        if not finite_points(wide_xy) or not (0.0 <= wide_xy[0] < wide_width and 0.0 <= wide_xy[1] < wide_height):
            continue
        pixel = [int(round(wide_xy[0])), int(round(wide_xy[1]))]
        pixel[0] = min(max(pixel[0], 0), wide_width - 1)
        pixel[1] = min(max(pixel[1], 0), wide_height - 1)
        prediction = pose_for_pixel(np.asarray(pixel, dtype=np.float32), h_wide_to_lattice, surface)
        samples.append(
            {
                "sample_index": len(samples),
                "wide_pixel": pixel,
                **prediction,
            }
        )
    if len(samples) != count:
        raise RuntimeError(f"Chỉ tạo được {len(samples)}/{count} điểm test nằm trong ảnh Wide.")
    return samples


def write_center_pairs_csv(path: Path, selected: list[dict[str, Any]]) -> None:
    columns = [
        "ptz_index", "row", "column", "wide_x", "wide_y", "pan", "tilt", "zoom",
        "wide_id", "inliers", "inlier_ratio", "median_reprojection_error_px", "center_prior_error_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for item in selected:
            center = item["mapped_ptz_center_on_wide"]
            state = item["actual_after_capture"]
            writer.writerow(
                {
                    "ptz_index": item["ptz_index"],
                    "row": item["row"],
                    "column": item["column"],
                    "wide_x": center[0],
                    "wide_y": center[1],
                    "pan": state["pan"],
                    "tilt": state["tilt"],
                    "zoom": state["zoom"],
                    "wide_id": item["wide_id"],
                    "inliers": item["inliers"],
                    "inlier_ratio": item["inlier_ratio"],
                    "median_reprojection_error_px": item["median_reprojection_error_px"],
                    "center_prior_error_px": item["center_prior_error_px"],
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="Wide/PTZ capture run; defaults to newest complete run.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to 02_aliked_lightglue_wide_ptz_mapping in the run.")
    parser.add_argument("--max-side", type=int, default=960, help="Working max image side for ALIKED.")
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--ransac-threshold", type=float, default=7.0, help="MAGSAC threshold in original-image pixels.")
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.18)
    parser.add_argument("--max-median-reprojection-error", type=float, default=7.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=1.2)
    parser.add_argument("--max-center-prior-error", type=float, default=650.0)
    parser.add_argument(
        "--center-boundary-margin",
        type=float,
        default=200.0,
        help="Accept a mapped PTZ centre slightly outside the Wide boundary when all other geometric checks pass.",
    )
    parser.add_argument("--lattice-ransac-threshold", type=float, default=0.45)
    parser.add_argument("--pchip-trigger", type=float, default=0.0015, help="Use PCHIP if affine pan/tilt RMSE exceeds this value.")
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA không khả dụng.")
    if args.max_side < 128 or args.max_keypoints < 32:
        raise SystemExit("--max-side và --max-keypoints quá nhỏ.")
    if args.sample_count < 1:
        raise SystemExit("--sample-count phải lớn hơn 0.")

    run_root = (args.run_root or find_latest_complete_run()).resolve()
    output_dir = (args.output_dir or run_root / "02_aliked_lightglue_wide_ptz_mapping").resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = output_dir / "features"
    match_dir = output_dir / "matches"
    diagnostic_dir = output_dir / "diagnostics"
    for directory in (feature_dir, match_dir, diagnostic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "mapping_run_metadata.json"
    manifest: dict[str, Any] = {
        "pipeline": "ALIKED + LightGlue + MAGSAC RANSAC + Wide-to-PTZ PCHIP",
        "status": "running",
        "started_at_utc": utc_now(),
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "arguments": vars(args),
        "stages": [],
    }
    write_json(manifest_path, manifest)

    try:
        wide_items, captures, ptz_metadata = load_run(run_root)
        device = torch.device(args.device)
        print(f"Loading ALIKED + LightGlue on {device}...", flush=True)
        extractor = KF.ALIKED.from_pretrained("aliked-n16", max_num_keypoints=args.max_keypoints, device=device).eval()
        matcher = KF.LightGlue("aliked").to(device).eval()

        wide_features: list[FeatureRecord] = []
        print(f"Extracting features: {len(wide_items)} Wide frame(s)...", flush=True)
        for index, item in enumerate(wide_items):
            path = run_root / item["image"]
            record = extract_features(f"wide_{index:02d}", path, extractor, args.max_side, device, feature_dir)
            wide_features.append(record)
            print(f"  Wide {index + 1}/{len(wide_items)}: {record.keypoints_original.shape[0]} keypoints", flush=True)

        ptz_features: dict[int, FeatureRecord] = {}
        print(f"Extracting features: {len(captures)} PTZ frame(s)...", flush=True)
        for number, capture in enumerate(captures, start=1):
            index = int(capture["index"])
            path = run_root / capture["raw_image"]
            record = extract_features(f"ptz_{index:02d}", path, extractor, args.max_side, device, feature_dir)
            ptz_features[index] = record
            print(f"  PTZ {number}/{len(captures)} (index {index}): {record.keypoints_original.shape[0]} keypoints", flush=True)

        manifest["stages"].append({"name": "ALIKED feature extraction", "status": "complete"})
        write_json(manifest_path, manifest)

        candidate_results: list[dict[str, Any]] = []
        selected: list[dict[str, Any]] = []
        print(f"Matching {len(captures)} PTZ frames against {len(wide_features)} Wide frames...", flush=True)
        for capture_number, capture in enumerate(captures, start=1):
            ptz = ptz_features[int(capture["index"])]
            candidates_for_ptz: list[dict[str, Any]] = []
            details_by_wide: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for wide in wide_features:
                result, wide_points, ptz_points, mask, _ = homography_candidate(
                    wide=wide,
                    ptz=ptz,
                    capture=capture,
                    matcher=matcher,
                    device=device,
                    args=args,
                    match_dir=match_dir,
                    run_root=run_root,
                )
                result.update(
                    {
                        "row": int(capture["row"]),
                        "column": int(capture["column"]),
                        "actual_after_capture": capture["actual_after_capture"],
                        "raw_image": capture["raw_image"],
                    }
                )
                candidate_results.append(result)
                candidates_for_ptz.append(result)
                details_by_wide[wide.identifier] = (wide_points, ptz_points, mask)

            plausible = [item for item in candidates_for_ptz if item["plausible"]]
            if plausible:
                winner = max(plausible, key=lambda item: item["selection_score"])
                winner = {**winner, "selected": True}
                selected.append(winner)
                wide = next(item for item in wide_features if item.identifier == winner["wide_id"])
                matched_wide, matched_ptz, mask = details_by_wide[winner["wide_id"]]
                save_match_visualization(
                    diagnostic_dir / f"selected_ptz_{winner['ptz_index']:02d}.jpg",
                    wide.image_path,
                    ptz.image_path,
                    matched_wide,
                    matched_ptz,
                    mask,
                )
                print(
                    f"  PTZ {capture_number:02d}/{len(captures)}: selected {winner['wide_id']} "
                    f"({winner['inliers']} inliers, centre prior "
                    f"{('disabled' if winner['center_prior_error_px'] is None else f'{winner['center_prior_error_px']:.1f}px')})",
                    flush=True,
                )
            else:
                best = max(candidates_for_ptz, key=lambda item: item["selection_score"])
                print(
                    f"  PTZ {capture_number:02d}/{len(captures)}: no plausible H "
                    f"(best {best['wide_id']}, {best['inliers']} inliers)",
                    flush=True,
                )

        all_pairs_path = output_dir / "pair_homographies.json"
        selected_path = output_dir / "selected_homographies.json"
        write_json(all_pairs_path, {"pairs": candidate_results})
        write_json(selected_path, {"selected": selected})
        manifest["stages"].append(
            {
                "name": "LightGlue matching and per-pair MAGSAC homographies",
                "status": "complete",
                "wide_ptz_pairs": len(candidate_results),
                "selected_ptz_pairs": len(selected),
            }
        )
        write_json(manifest_path, manifest)

        if len(selected) < 8:
            raise RuntimeError(f"Chỉ có {len(selected)} cặp H đáng tin; cần tối thiểu 8 để fit H tổng quát.")

        h_wide_to_lattice, global_inlier_mask, centers, lattice, lattice_errors = fit_homography_wide_to_lattice(selected, args)
        global_inliers = [item for item, is_inlier in zip(selected, global_inlier_mask, strict=True) if is_inlier]
        unique_rows = {int(item["row"]) for item in global_inliers}
        unique_columns = {int(item["column"]) for item in global_inliers}
        if len(global_inliers) < 8 or len(unique_rows) < 3 or len(unique_columns) < 3:
            raise RuntimeError(
                "H tổng quát không có đủ inlier phân bố theo grid "
                f"({len(global_inliers)} points, {len(unique_rows)} rows, {len(unique_columns)} columns)."
            )

        grid, u_values, v_values = state_grid(captures)
        probe = linearity_probe(grid)
        use_pchip = max(probe["affine_rmse"]["pan"], probe["affine_rmse"]["tilt"]) > args.pchip_trigger
        surface = PoseSurface(grid, use_pchip=use_pchip)
        global_inlier_errors = lattice_errors[global_inlier_mask]

        # Evaluate each selected centre using the global H and the chosen pose surface.
        centre_pairs: list[dict[str, Any]] = []
        pose_errors: list[np.ndarray] = []
        for item in selected:
            prediction = pose_for_pixel(
                np.asarray(item["mapped_ptz_center_on_wide"], dtype=np.float32),
                h_wide_to_lattice,
                surface,
            )
            actual = item["actual_after_capture"]
            error = np.asarray(
                [prediction["pan"] - float(actual["pan"]), prediction["tilt"] - float(actual["tilt"]), prediction["zoom"] - float(actual["zoom"])],
                dtype=np.float64,
            )
            pose_errors.append(error)
            centre_pairs.append({**item, "wide_to_pose_prediction": prediction, "pose_error": error.tolist()})
        pose_error_values = np.asarray(pose_errors, dtype=np.float64)

        wide_reference = wide_features[0]
        samples = make_random_predictions(
            h_wide_to_lattice=h_wide_to_lattice,
            surface=surface,
            wide_width=wide_reference.width,
            wide_height=wide_reference.height,
            count=args.sample_count,
            seed=args.seed,
        )
        make_center_overlay(diagnostic_dir / "mapped_ptz_centers_on_wide.jpg", wide_reference.image_path, selected)
        make_sample_overlay(diagnostic_dir / "random_wide_predictions.jpg", wide_reference.image_path, samples)
        write_center_pairs_csv(output_dir / "ptz_center_to_wide_pairs.csv", selected)

        global_mapping = {
            "model": {
                "H_wide_pixel_to_ptz_lattice": h_wide_to_lattice.tolist(),
                "H_ptz_lattice_to_wide_pixel": np.linalg.inv(h_wide_to_lattice).tolist(),
                "state_surface": "pchip_2d_separable" if use_pchip else "affine",
                "pchip_used": use_pchip,
                "pchip_trigger": args.pchip_trigger,
                "lattice_definition": {
                    "u": "PTZ grid column, 0 through 4",
                    "v": "PTZ grid row, 0 through 5",
                    "columns": int(len(u_values)),
                    "rows": int(len(v_values)),
                },
                "state_grid_actual_after_capture": grid.tolist(),
                "linearity_probe": probe,
            },
            "fit_quality": {
                "selected_pair_count": len(selected),
                "global_h_inlier_count": len(global_inliers),
                "global_h_inlier_rows": sorted(unique_rows),
                "global_h_inlier_columns": sorted(unique_columns),
                "lattice_error_px_units": {
                    "mean": float(np.mean(global_inlier_errors)),
                    "median": float(np.median(global_inlier_errors)),
                    "max": float(np.max(global_inlier_errors)),
                },
                "centre_pose_error_actual_minus_prediction": {
                    "mean_abs": {"pan": float(np.mean(np.abs(pose_error_values[:, 0]))), "tilt": float(np.mean(np.abs(pose_error_values[:, 1]))), "zoom": float(np.mean(np.abs(pose_error_values[:, 2])))},
                    "max_abs": {"pan": float(np.max(np.abs(pose_error_values[:, 0]))), "tilt": float(np.max(np.abs(pose_error_values[:, 1]))), "zoom": float(np.max(np.abs(pose_error_values[:, 2])))},
                },
                "note": "H is a planar image relation. PCHIP maps its lattice coordinates to recorded PTZ state; predictions outside the lattice are clamped to its nearest calibrated edge.",
            },
            "centre_pairs": centre_pairs,
            "random_wide_pixel_predictions": samples,
        }
        write_json(output_dir / "wide_pixel_to_ptz_mapping.json", global_mapping)
        write_json(output_dir / "random_30_wide_pixel_predictions.json", {"samples": samples})

        manifest["stages"].append(
            {
                "name": "global Wide-to-lattice H and pose surface",
                "status": "complete",
                "pchip_used": use_pchip,
                "global_h_inliers": len(global_inliers),
                "prediction_count": len(samples),
            }
        )
        manifest["status"] = "complete"
        return_code = 0
    except Exception as exc:  # noqa: BLE001 - preserve complete failure context in output metadata
        manifest["status"] = "failed"
        manifest["error_type"] = type(exc).__name__
        manifest["error"] = str(exc)
        return_code = 1
    finally:
        manifest["finished_at_utc"] = utc_now()
        write_json(manifest_path, manifest)

    print(json.dumps({"status": manifest["status"], "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
