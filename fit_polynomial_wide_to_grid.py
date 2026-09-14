#!/usr/bin/env python3
"""Fit a robust Polynomial Wide-pixel -> PTZ-grid model.

This is an experiment stage. It keeps the 30 local PTZ->Wide homographies and
replaces only the global Wide->grid homography. The following grid->pose
surface is reused from the selected mapping run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator


PIPELINE_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def term_powers(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total, -1, -1)]


def normalize_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = np.asarray([max(width / 2.0, 1.0), max(height / 2.0, 1.0)], dtype=np.float64)
    centre = np.asarray([width / 2.0, height / 2.0], dtype=np.float64)
    return (np.asarray(points, dtype=np.float64) - centre) / scale


def design_matrix(points_normalized: np.ndarray, powers: list[tuple[int, int]]) -> np.ndarray:
    x = points_normalized[:, 0]
    y = points_normalized[:, 1]
    return np.column_stack([(x**i) * (y**j) for i, j in powers])


def fit_coefficients(features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, int]:
    coefficients, _, rank, _ = np.linalg.lstsq(features, targets, rcond=None)
    return coefficients, int(rank)


def predict(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return features @ coefficients


def residuals(features: np.ndarray, coefficients: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.linalg.norm(predict(features, coefficients) - targets, axis=1)


def robust_polynomial_fit(
    points_normalized: np.ndarray,
    targets: np.ndarray,
    degree: int,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    powers = term_powers(degree)
    features = design_matrix(points_normalized, powers)
    sample_size = len(powers)
    if len(targets) < sample_size:
        raise RuntimeError(f"Bậc {degree} cần ít nhất {sample_size} điểm, chỉ có {len(targets)}.")

    rng = np.random.default_rng(seed + degree)
    best: tuple[int, float, np.ndarray, np.ndarray] | None = None
    for _ in range(iterations):
        sample = rng.choice(len(targets), size=sample_size, replace=False)
        coefficients, rank = fit_coefficients(features[sample], targets[sample])
        if rank < sample_size:
            continue
        errors = residuals(features, coefficients, targets)
        mask = errors <= threshold
        count = int(mask.sum())
        score_rmse = float(np.sqrt(np.mean(errors[mask] ** 2))) if count else math.inf
        if best is None or count > best[0] or (count == best[0] and score_rmse < best[1]):
            best = (count, score_rmse, mask, coefficients)

    if best is None:
        coefficients, rank = fit_coefficients(features, targets)
        if rank < sample_size:
            raise RuntimeError(f"Không fit được Polynomial bậc {degree}: ma trận suy biến.")
        mask = np.ones(len(targets), dtype=bool)
    else:
        mask = best[2]
        if int(mask.sum()) < sample_size:
            mask = np.ones(len(targets), dtype=bool)
        coefficients, rank = fit_coefficients(features[mask], targets[mask])
        if rank < sample_size:
            mask = np.ones(len(targets), dtype=bool)
            coefficients, rank = fit_coefficients(features, targets)
        if rank < sample_size:
            raise RuntimeError(f"Không refit được Polynomial bậc {degree}: ma trận suy biến.")

    # A few deterministic robust-refinement passes reduce the effect of points
    # just outside the initial RANSAC threshold.
    for _ in range(3):
        errors = residuals(features, coefficients, targets)
        refined_mask = errors <= threshold
        if int(refined_mask.sum()) < sample_size:
            break
        coefficients, rank = fit_coefficients(features[refined_mask], targets[refined_mask])
        if rank < sample_size:
            break
        mask = refined_mask

    errors = residuals(features, coefficients, targets)
    return {
        "degree": degree,
        "powers": powers,
        "coefficients": coefficients,
        "features": features,
        "inlier_mask": mask,
        "errors": errors,
        "inlier_count": int(mask.sum()),
        "rmse_all": float(np.sqrt(np.mean(errors**2))),
        "rmse_inliers": float(np.sqrt(np.mean(errors[mask] ** 2))),
        "median_all": float(np.median(errors)),
        "max_all": float(np.max(errors)),
    }


def leave_one_out_rmse(model: dict[str, Any], points_normalized: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    degree = int(model["degree"])
    powers = term_powers(degree)
    features = design_matrix(points_normalized, powers)
    inlier_indices = np.flatnonzero(model["inlier_mask"])
    all_errors: list[float] = []
    inlier_errors: list[float] = []
    for index in inlier_indices:
        train_indices = inlier_indices[inlier_indices != index]
        coefficients, rank = fit_coefficients(features[train_indices], targets[train_indices])
        if rank < len(powers):
            continue
        error = float(np.linalg.norm(predict(features[index : index + 1], coefficients)[0] - targets[index]))
        inlier_errors.append(error)
    for index in range(len(targets)):
        train_indices = np.asarray([item for item in range(len(targets)) if item != index], dtype=np.int64)
        coefficients, rank = fit_coefficients(features[train_indices], targets[train_indices])
        if rank < len(powers):
            continue
        all_errors.append(float(np.linalg.norm(predict(features[index : index + 1], coefficients)[0] - targets[index])))
    return (
        float(np.sqrt(np.mean(np.asarray(all_errors) ** 2))) if all_errors else math.inf,
        float(np.sqrt(np.mean(np.asarray(inlier_errors) ** 2))) if inlier_errors else math.inf,
    )


class PoseSurface:
    def __init__(self, grid: np.ndarray, method: str, affine_coefficients: np.ndarray) -> None:
        self.grid = grid
        self.method = method
        self.affine_coefficients = affine_coefficients
        self.rows, self.columns, _ = grid.shape
        self.u_values = np.arange(self.columns, dtype=np.float64)
        self.v_values = np.arange(self.rows, dtype=np.float64)

    def predict(self, u: float, v: float) -> np.ndarray:
        if self.method == "affine":
            return np.asarray([1.0, u, v], dtype=np.float64) @ self.affine_coefficients
        result = np.empty(3, dtype=np.float64)
        for axis in range(3):
            across_rows = np.asarray(
                [float(PchipInterpolator(self.u_values, self.grid[row, :, axis], extrapolate=True)(u)) for row in range(self.rows)],
                dtype=np.float64,
            )
            result[axis] = float(PchipInterpolator(self.v_values, across_rows, extrapolate=True)(v))
        return result


def pose_from_grid(grid_uv: np.ndarray, surface: PoseSurface, clamp: bool = True) -> dict[str, Any]:
    raw_u, raw_v = float(grid_uv[0]), float(grid_uv[1])
    used_u, used_v = raw_u, raw_v
    clamped = False
    if clamp:
        used_u = min(max(used_u, 0.0), surface.columns - 1.0)
        used_v = min(max(used_v, 0.0), surface.rows - 1.0)
        clamped = not (math.isclose(raw_u, used_u) and math.isclose(raw_v, used_v))
    state = surface.predict(used_u, used_v)
    return {
        "lattice_uv_raw": [raw_u, raw_v],
        "lattice_uv_used": [used_u, used_v],
        "clamped": clamped,
        "pan": float(state[0]),
        "tilt": float(state[1]),
        "zoom": float(state[2]),
    }


def draw_fit_overlay(path: Path, image_path: Path, centres: np.ndarray, errors: np.ndarray, inlier_mask: np.ndarray) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được ảnh Wide: {image_path}")
    for index, (point, error, inlier) in enumerate(zip(centres, errors, inlier_mask, strict=True)):
        xy = tuple(np.round(point).astype(int))
        color = (0, 210, 0) if inlier else (0, 0, 255)
        cv2.drawMarker(image, xy, color, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
        cv2.putText(image, f"{index:02d}:{error:.2f}", (xy[0] + 8, xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(image, "Polynomial Wide -> grid | green=inlier red=outlier | error in grid units", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def draw_grid_comparison(path: Path, targets: np.ndarray, predicted: np.ndarray) -> None:
    canvas = np.full((360, 1000, 3), 245, dtype=np.uint8)
    left, right, top, bottom = 70, 940, 70, 300
    max_u = max(float(np.max(targets[:, 0])), 1.0)
    max_v = max(float(np.max(targets[:, 1])), 1.0)

    def xy(point: np.ndarray) -> tuple[int, int]:
        return (int(round(left + point[0] * (right - left) / max_u)), int(round(top + point[1] * (bottom - top) / max_v)))

    for row in range(int(max_v) + 1):
        y = xy(np.asarray([0.0, float(row)]))[1]
        cv2.line(canvas, (left, y), (right, y), (215, 215, 215), 1)
        cv2.putText(canvas, str(row), (35, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    for column in range(int(max_u) + 1):
        x = xy(np.asarray([float(column), 0.0]))[0]
        cv2.line(canvas, (x, top), (x, bottom), (215, 215, 215), 1)
        cv2.putText(canvas, str(column), (x - 8, bottom + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    for index, (target, estimate) in enumerate(zip(targets, predicted, strict=True)):
        target_xy, estimate_xy = xy(target), xy(estimate)
        cv2.line(canvas, target_xy, estimate_xy, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(canvas, target_xy, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.drawMarker(canvas, estimate_xy, (255, 0, 0), cv2.MARKER_CROSS, 12, 2, cv2.LINE_AA)
        if index % 5 == 0:
            cv2.putText(canvas, str(index), (estimate_xy[0] + 7, estimate_xy[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Grid target (black dot) vs Polynomial prediction (blue cross)", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-degree", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--ransac-threshold", type=float, default=0.35, help="Grid-coordinate residual threshold.")
    parser.add_argument("--ransac-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    source_mapping_path = run_root / "02_aliked_lightglue_wide_ptz_mapping" / "wide_pixel_to_ptz_mapping.json"
    source_mapping = json.loads(source_mapping_path.read_text(encoding="utf-8"))
    pairs = sorted(source_mapping.get("centre_pairs", []), key=lambda item: int(item["ptz_index"]))
    if len(pairs) != 30:
        raise RuntimeError(f"Cần 30 centre pairs, hiện có {len(pairs)}.")

    wide_path = run_root / pairs[0]["wide_image"]
    wide_image = cv2.imread(str(wide_path), cv2.IMREAD_COLOR)
    if wide_image is None:
        raise RuntimeError(f"Không đọc được Wide image: {wide_path}")
    height, width = wide_image.shape[:2]
    centres = np.asarray([item["mapped_ptz_center_on_wide"] for item in pairs], dtype=np.float64)
    targets = np.asarray([[item["column"], item["row"]] for item in pairs], dtype=np.float64)
    centres_normalized = normalize_points(centres, width, height)

    output_dir = (args.output_dir or run_root / f"03_polynomial_grid_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir()

    baseline_h = np.asarray(source_mapping["model"]["H_wide_pixel_to_ptz_lattice"], dtype=np.float64)
    baseline_predicted = cv2.perspectiveTransform(centres.astype(np.float32).reshape(-1, 1, 2), baseline_h).reshape(-1, 2).astype(np.float64)
    baseline_errors = np.linalg.norm(baseline_predicted - targets, axis=1)

    models: list[dict[str, Any]] = []
    for degree in range(1, args.max_degree + 1):
        model = robust_polynomial_fit(centres_normalized, targets, degree, args.ransac_threshold, args.ransac_iterations, args.seed)
        loo_all, loo_inliers = leave_one_out_rmse(model, centres_normalized, targets)
        model["loo_rmse_all"] = loo_all
        model["loo_rmse_inliers"] = loo_inliers
        models.append(model)

    winner = min(models, key=lambda item: (float(item["loo_rmse_inliers"]), int(item["degree"])))
    powers = winner["powers"]
    winner_predicted = predict(winner["features"], winner["coefficients"])

    source_model = source_mapping["model"]
    grid = np.asarray(source_model["state_grid_actual_after_capture"], dtype=np.float64)
    affine_coefficients = np.asarray(source_model["linearity_probe"]["affine_coefficients_intercept_u_v"], dtype=np.float64)
    surface = PoseSurface(grid, source_model["state_surface"], affine_coefficients)

    rows: list[dict[str, Any]] = []
    centre_predictions: list[dict[str, Any]] = []
    for index, (pair, target, estimate) in enumerate(zip(pairs, targets, winner_predicted, strict=True)):
        pose = pose_from_grid(estimate, surface)
        actual_state = pair["actual_after_capture"]
        rows.append(
            {
                "ptz_index": int(pair["ptz_index"]),
                "row": int(pair["row"]),
                "column": int(pair["column"]),
                "wide_x": float(centres[index, 0]),
                "wide_y": float(centres[index, 1]),
                "grid_u_target": float(target[0]),
                "grid_v_target": float(target[1]),
                "grid_u_polynomial": float(estimate[0]),
                "grid_v_polynomial": float(estimate[1]),
                "polynomial_grid_error": float(np.linalg.norm(estimate - target)),
                "homography_grid_error": float(baseline_errors[index]),
                "polynomial_inlier": bool(winner["inlier_mask"][index]),
                "predicted_pan": pose["pan"],
                "predicted_tilt": pose["tilt"],
                "predicted_zoom": pose["zoom"],
                "actual_pan": float(actual_state["pan"]),
                "actual_tilt": float(actual_state["tilt"]),
                "actual_zoom": float(actual_state["zoom"]),
            }
        )
        centre_predictions.append({**rows[-1], "lattice_uv_raw": pose["lattice_uv_raw"], "lattice_uv_used": pose["lattice_uv_used"], "clamped": pose["clamped"]})

    draw_fit_overlay(diagnostic_dir / "polynomial_grid_fit_on_wide.jpg", wide_path, centres, winner["errors"], winner["inlier_mask"])
    draw_grid_comparison(diagnostic_dir / "polynomial_grid_target_vs_prediction.jpg", targets, winner_predicted)

    comparison_path = output_dir / "polynomial_grid_fit.csv"
    with comparison_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model_reports = []
    for model in models:
        model_reports.append(
            {
                "degree": int(model["degree"]),
                "terms": [f"x^{i} y^{j}" for i, j in model["powers"]],
                "inlier_count": int(model["inlier_count"]),
                "rmse_all_grid_units": float(model["rmse_all"]),
                "rmse_inliers_grid_units": float(model["rmse_inliers"]),
                "median_all_grid_units": float(model["median_all"]),
                "max_all_grid_units": float(model["max_all"]),
                "loo_rmse_all_grid_units": float(model["loo_rmse_all"]),
                "loo_rmse_inliers_grid_units": float(model["loo_rmse_inliers"]),
            }
        )

    polynomial_mapping = {
        "status": "complete",
        "pipeline": "local H_i unchanged + robust Polynomial Wide pixel -> PTZ grid + existing pose surface",
        "source_run": str(run_root),
        "source_mapping": str(source_mapping_path),
        "model": {
            "mapping_type": "polynomial_wide_pixel_to_ptz_lattice",
            "degree": int(winner["degree"]),
            "terms_powers_xy": [[int(i), int(j)] for i, j in powers],
            "coefficients_uv": np.asarray(winner["coefficients"], dtype=np.float64).tolist(),
            "input_normalization": {
                "x": "(pixel_x - width/2) / (width/2)",
                "y": "(pixel_y - height/2) / (height/2)",
                "width": width,
                "height": height,
            },
            "lattice_definition": {
                "u": f"PTZ grid column 0..{int(np.max(targets[:, 0]))}",
                "v": f"PTZ grid row 0..{int(np.max(targets[:, 1]))}",
                "columns": int(np.max(targets[:, 0])) + 1,
                "rows": int(np.max(targets[:, 1])) + 1,
            },
            "state_surface": source_model["state_surface"],
            "state_grid_actual_after_capture": grid.tolist(),
            "linearity_probe": source_model["linearity_probe"],
        },
        "fit_quality": {
            "selected_degree": int(winner["degree"]),
            "polynomial_grid_rmse_all": float(winner["rmse_all"]),
            "polynomial_grid_rmse_inliers": float(winner["rmse_inliers"]),
            "polynomial_grid_median_all": float(winner["median_all"]),
            "polynomial_grid_max_all": float(winner["max_all"]),
            "polynomial_grid_loo_rmse_all": float(winner["loo_rmse_all"]),
            "polynomial_grid_loo_rmse_inliers": float(winner["loo_rmse_inliers"]),
            "polynomial_inlier_count": int(winner["inlier_count"]),
            "baseline_homography_grid_rmse": float(np.sqrt(np.mean(baseline_errors**2))),
            "baseline_homography_grid_median": float(np.median(baseline_errors)),
            "baseline_homography_grid_max": float(np.max(baseline_errors)),
            "model_comparison": model_reports,
            "ransac_threshold_grid_units": args.ransac_threshold,
        },
        "centre_predictions": centre_predictions,
    }
    write_json(output_dir / "polynomial_wide_pixel_to_ptz_mapping.json", polynomial_mapping)
    write_json(output_dir / "model_comparison.json", {"models": model_reports})
    (output_dir / "README.md").write_text(
        "# Polynomial Wide-to-grid experiment\n\n"
        "Thử nghiệm này giữ nguyên 30 homography cục bộ `H_i` dùng để lấy tâm PTZ trên Wide.\n"
        "Chỉ thay tầng toàn cục `Wide pixel -> PTZ grid` bằng Polynomial robust.\n\n"
        f"- Bậc được chọn: {int(winner['degree'])}.\n"
        f"- Inlier: {int(winner['inlier_count'])}/30.\n"
        f"- Polynomial RMSE: {float(winner['rmse_all']):.6f} grid units.\n"
        f"- Homography baseline RMSE: {float(np.sqrt(np.mean(baseline_errors**2))):.6f} grid units.\n"
        "- `polynomial_wide_pixel_to_ptz_mapping.json`: mô hình Polynomial và tầng pose surface.\n"
        "- `polynomial_grid_fit.csv`: so sánh tọa độ grid dự đoán với grid thật cho 30 tâm.\n"
        "- `diagnostics/polynomial_grid_fit_on_wide.jpg`: tâm Wide, màu xanh là inlier, đỏ là outlier.\n"
        "- `diagnostics/polynomial_grid_target_vs_prediction.jpg`: điểm grid thật và dự đoán.\n\n"
        "Polynomial chỉ thay mô hình Wide -> grid; chưa điều khiển camera và chưa thay 30 H_i cục bộ.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "selected_degree": int(winner["degree"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
