#!/usr/bin/env python3
"""Experiment with Polynomial models at both calibration stages.

Stage 1 replaces each local PTZ->Wide homography with a robust degree-2
Polynomial fitted to the saved LightGlue matches. Stage 2 fits a robust
Polynomial Wide->grid model from the 30 newly estimated PTZ centres.
No camera connection or movement is performed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from fit_polynomial_wide_to_grid import (
    PoseSurface,
    design_matrix,
    leave_one_out_rmse,
    normalize_points,
    pose_from_grid,
    robust_polynomial_fit,
    term_powers,
)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def fit_global_models(points: np.ndarray, targets: np.ndarray, width: int, height: int, threshold: float, seed: int) -> list[dict[str, Any]]:
    normalized = normalize_points(points, width, height)
    models: list[dict[str, Any]] = []
    for degree in (1, 2, 3):
        model = robust_polynomial_fit(normalized, targets, degree, threshold, 3000, seed)
        loo_all, loo_inliers = leave_one_out_rmse(model, normalized, targets)
        model["loo_rmse_all"] = loo_all
        model["loo_rmse_inliers"] = loo_inliers
        models.append(model)
    return models


def choose_global_model(models: list[dict[str, Any]]) -> dict[str, Any]:
    return min(models, key=lambda item: (float(item["loo_rmse_inliers"]), int(item["degree"])))


def polynomial_report(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "degree": int(model["degree"]),
        "terms": [[int(i), int(j)] for i, j in model["powers"]],
        "inlier_count": int(model["inlier_count"]),
        "rmse_all": float(model["rmse_all"]),
        "rmse_inliers": float(model["rmse_inliers"]),
        "median_all": float(model["median_all"]),
        "max_all": float(model["max_all"]),
        "loo_rmse_all": float(model["loo_rmse_all"]),
        "loo_rmse_inliers": float(model["loo_rmse_inliers"]),
    }


def fit_homography(points: np.ndarray, targets: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix, mask = cv2.findHomography(points.astype(np.float32), targets.astype(np.float32), cv2.USAC_MAGSAC, threshold, maxIters=10000, confidence=0.999)
    if matrix is None or mask is None:
        raise RuntimeError("Không fit được homography Wide -> grid trong experiment.")
    matrix = matrix / matrix[2, 2]
    predicted = cv2.perspectiveTransform(points.astype(np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.float64)
    return matrix, mask.ravel().astype(bool), predicted


def draw_centres(path: Path, image_path: Path, h_centres: np.ndarray, polynomial_centres: np.ndarray) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được Wide reference: {image_path}")
    for index, (h_point, p_point) in enumerate(zip(h_centres, polynomial_centres, strict=True)):
        h_xy = tuple(np.round(h_point).astype(int))
        p_xy = tuple(np.round(p_point).astype(int))
        cv2.circle(image, p_xy, 9, (0, 220, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(image, h_xy, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 22, 2, cv2.LINE_AA)
        cv2.line(image, h_xy, p_xy, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, str(index), (p_xy[0] + 8, p_xy[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.putText(image, "Local H centre (orange) vs local Polynomial centre (green)", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def draw_grid_comparison(path: Path, targets: np.ndarray, predicted: np.ndarray) -> None:
    canvas = np.full((360, 1000, 3), 245, dtype=np.uint8)
    left, right, top, bottom = 70, 940, 70, 300
    max_u = max(float(np.max(targets[:, 0])), 1.0)
    max_v = max(float(np.max(targets[:, 1])), 1.0)

    def xy(point: np.ndarray) -> tuple[int, int]:
        return int(round(left + point[0] * (right - left) / max_u)), int(round(top + point[1] * (bottom - top) / max_v))

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
    cv2.putText(canvas, "Grid target (black) vs Polynomial prediction (blue)", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2, cv2.LINE_AA)
    cv2.imwrite(str(path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-degree", type=int, default=2, choices=(2,))
    parser.add_argument("--local-ransac-threshold-px", type=float, default=7.0)
    parser.add_argument("--global-ransac-threshold-grid", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260914)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    source_mapping_path = run_root / "02_aliked_lightglue_wide_ptz_mapping" / "wide_pixel_to_ptz_mapping.json"
    source_mapping = json.loads(source_mapping_path.read_text(encoding="utf-8"))
    pairs = sorted(source_mapping.get("centre_pairs", []), key=lambda item: int(item["ptz_index"]))
    if len(pairs) != 30:
        raise RuntimeError(f"Cần đúng 30 cặp PTZ–Wide, hiện có {len(pairs)}.")

    reference_wide_path = run_root / pairs[0]["wide_image"]
    reference_wide = cv2.imread(str(reference_wide_path), cv2.IMREAD_COLOR)
    if reference_wide is None:
        raise RuntimeError(f"Không đọc được Wide reference: {reference_wide_path}")
    wide_height, wide_width = reference_wide.shape[:2]

    h_centres = np.asarray([item["mapped_ptz_center_on_wide"] for item in pairs], dtype=np.float64)
    lattice_targets = np.asarray([[item["column"], item["row"]] for item in pairs], dtype=np.float64)
    local_results: list[dict[str, Any]] = []
    polynomial_centres: list[list[float]] = []
    for pair in pairs:
        match_path = run_root / pair["match_file"]
        match_data = np.load(match_path)
        ptz_points = np.asarray(match_data["ptz_points_xy"], dtype=np.float64)
        wide_points = np.asarray(match_data["wide_points_xy"], dtype=np.float64)
        ptz_image = cv2.imread(str(run_root / pair["ptz_image"]), cv2.IMREAD_COLOR)
        if ptz_image is None:
            raise RuntimeError(f"Không đọc được PTZ image: {run_root / pair['ptz_image']}")
        ptz_height, ptz_width = ptz_image.shape[:2]
        normalized_ptz = normalize_points(ptz_points, ptz_width, ptz_height)
        local_model = robust_polynomial_fit(
            normalized_ptz,
            wide_points,
            args.local_degree,
            args.local_ransac_threshold_px,
            3000,
            args.seed + int(pair["ptz_index"]),
        )
        centre_input = normalize_points(np.asarray([[ptz_width / 2.0, ptz_height / 2.0]], dtype=np.float64), ptz_width, ptz_height)
        centre_features = design_matrix(centre_input, local_model["powers"])
        polynomial_centre = (centre_features @ local_model["coefficients"])[0]
        polynomial_centres.append([float(polynomial_centre[0]), float(polynomial_centre[1])])

        h_matrix = np.asarray(pair["H_ptz_to_wide"], dtype=np.float64)
        h_prediction = cv2.perspectiveTransform(ptz_points.astype(np.float32).reshape(-1, 1, 2), h_matrix).reshape(-1, 2).astype(np.float64)
        h_errors = np.linalg.norm(h_prediction - wide_points, axis=1)
        saved_h_mask = np.asarray(match_data["inlier_mask"], dtype=bool)
        local_results.append(
            {
                "ptz_index": int(pair["ptz_index"]),
                "row": int(pair["row"]),
                "column": int(pair["column"]),
                "wide_id": pair["wide_id"],
                "match_file": pair["match_file"],
                "ptz_image": pair["ptz_image"],
                "match_count": int(len(ptz_points)),
                "saved_h_inlier_count": int(saved_h_mask.sum()),
                "local_polynomial_degree": args.local_degree,
                "local_polynomial_inlier_count": int(local_model["inlier_count"]),
                "local_polynomial_rmse_all_px": float(local_model["rmse_all"]),
                "local_polynomial_rmse_inliers_px": float(local_model["rmse_inliers"]),
                "local_polynomial_median_all_px": float(local_model["median_all"]),
                "local_polynomial_max_all_px": float(local_model["max_all"]),
                "local_polynomial_centre_on_wide": [float(polynomial_centre[0]), float(polynomial_centre[1])],
                "saved_h_centre_on_wide": [
                    float(h_centres[len(local_results), 0]),
                    float(h_centres[len(local_results), 1]),
                ],
                "centre_difference_polynomial_minus_h_px": [
                    float(polynomial_centre[0] - h_centres[len(local_results)][0]),
                    float(polynomial_centre[1] - h_centres[len(local_results)][1]),
                ],
                "h_rmse_all_px": float(np.sqrt(np.mean(h_errors**2))),
                "h_median_all_px": float(np.median(h_errors)),
                "h_max_all_px": float(np.max(h_errors)),
                "h_rmse_saved_inliers_px": float(np.sqrt(np.mean(h_errors[saved_h_mask] ** 2))) if saved_h_mask.any() else math.inf,
                "local_polynomial_terms_powers_xy": [[int(i), int(j)] for i, j in local_model["powers"]],
                "local_polynomial_coefficients_wide_xy": np.asarray(local_model["coefficients"], dtype=np.float64).tolist(),
            }
        )

    polynomial_centres_array = np.asarray(polynomial_centres, dtype=np.float64)
    source_h_global = np.asarray(source_mapping["model"]["H_wide_pixel_to_ptz_lattice"], dtype=np.float64)
    source_h_global_predicted = cv2.perspectiveTransform(h_centres.astype(np.float32).reshape(-1, 1, 2), source_h_global).reshape(-1, 2).astype(np.float64)
    source_h_global_errors = np.linalg.norm(source_h_global_predicted - lattice_targets, axis=1)

    source_global_models = fit_global_models(h_centres, lattice_targets, wide_width, wide_height, args.global_ransac_threshold_grid, args.seed)
    source_global_poly = choose_global_model(source_global_models)
    polynomial_global_models = fit_global_models(polynomial_centres_array, lattice_targets, wide_width, wide_height, args.global_ransac_threshold_grid, args.seed + 100)
    polynomial_global_poly = choose_global_model(polynomial_global_models)
    poly_centres_h, poly_centres_h_mask, poly_centres_h_predicted = fit_homography(polynomial_centres_array, lattice_targets, 0.45)
    poly_centres_h_errors = np.linalg.norm(poly_centres_h_predicted - lattice_targets, axis=1)

    source_surface_model = source_mapping["model"]
    state_grid = np.asarray(source_surface_model["state_grid_actual_after_capture"], dtype=np.float64)
    affine_coefficients = np.asarray(source_surface_model["linearity_probe"]["affine_coefficients_intercept_u_v"], dtype=np.float64)
    surface = PoseSurface(state_grid, source_surface_model["state_surface"], affine_coefficients)

    def global_poly_predict(model: dict[str, Any], points: np.ndarray) -> np.ndarray:
        features = design_matrix(normalize_points(points, wide_width, wide_height), model["powers"])
        return features @ model["coefficients"]

    source_poly_global_predicted = global_poly_predict(source_global_poly, h_centres)
    polynomial_poly_global_predicted = global_poly_predict(polynomial_global_poly, polynomial_centres_array)

    def pose_errors(grid_predictions: np.ndarray, source_centres: list[dict[str, Any]]) -> dict[str, Any]:
        errors: list[np.ndarray] = []
        for grid_uv, pair in zip(grid_predictions, source_centres, strict=True):
            pose = pose_from_grid(grid_uv, surface)
            actual = pair["actual_after_capture"]
            errors.append(np.asarray([pose["pan"] - float(actual["pan"]), pose["tilt"] - float(actual["tilt"]), pose["zoom"] - float(actual["zoom"])], dtype=np.float64))
        values = np.asarray(errors, dtype=np.float64)
        return {"mean_abs": np.mean(np.abs(values), axis=0).tolist(), "max_abs": np.max(np.abs(values), axis=0).tolist()}

    pose_quality = {
        "baseline_H_local_centres_plus_H_global": source_mapping["fit_quality"]["centre_pose_error_actual_minus_prediction"],
        "H_local_centres_plus_Polynomial_global": pose_errors(source_poly_global_predicted, pairs),
        "Polynomial_local_centres_plus_H_global": pose_errors(cv2.perspectiveTransform(polynomial_centres_array.astype(np.float32).reshape(-1, 1, 2), source_h_global).reshape(-1, 2), pairs),
        "Polynomial_local_centres_plus_Polynomial_global": pose_errors(polynomial_poly_global_predicted, pairs),
    }

    output_dir = (args.output_dir or run_root / f"04_polynomial_both_stages_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir = output_dir / "diagnostics"
    diagnostic_dir.mkdir()
    draw_centres(diagnostic_dir / "local_h_centres_vs_polynomial_centres.jpg", reference_wide_path, h_centres, polynomial_centres_array)
    draw_grid_comparison(diagnostic_dir / "both_stages_polynomial_grid_target_vs_prediction.jpg", lattice_targets, polynomial_poly_global_predicted)

    csv_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(local_results):
        csv_rows.append(
            {
                "ptz_index": pair["ptz_index"],
                "row": pair["row"],
                "column": pair["column"],
                "h_centre_x": h_centres[index, 0],
                "h_centre_y": h_centres[index, 1],
                "polynomial_centre_x": polynomial_centres_array[index, 0],
                "polynomial_centre_y": polynomial_centres_array[index, 1],
                "centre_delta_px": float(np.linalg.norm(polynomial_centres_array[index] - h_centres[index])),
                "h_rmse_all_px": pair["h_rmse_all_px"],
                "polynomial_rmse_all_px": pair["local_polynomial_rmse_all_px"],
                "h_inliers": pair["saved_h_inlier_count"],
                "polynomial_inliers": pair["local_polynomial_inlier_count"],
            }
        )
    with (output_dir / "local_pair_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    result = {
        "status": "complete",
        "pipeline": "Polynomial local PTZ->Wide centres + Polynomial global Wide->grid + existing pose surface",
        "source_run": str(run_root),
        "source_mapping": str(source_mapping_path),
        "local_stage": {
            "degree": args.local_degree,
            "input": "saved LightGlue match points in each PTZ/Wide pair",
            "output": "PTZ image centre projected to Wide",
            "ransac_threshold_px": args.local_ransac_threshold_px,
            "pairs": local_results,
            "mean_centre_delta_from_saved_H_px": float(np.mean(np.linalg.norm(polynomial_centres_array - h_centres, axis=1))),
            "max_centre_delta_from_saved_H_px": float(np.max(np.linalg.norm(polynomial_centres_array - h_centres, axis=1))),
        },
        "global_stage": {
            "input": "30 Polynomial local centres",
            "target": "5x6 PTZ grid coordinates (u=column, v=row)",
            "ransac_threshold_grid_units": args.global_ransac_threshold_grid,
            "selected_polynomial": polynomial_report(polynomial_global_poly),
            "all_polynomial_models": [polynomial_report(model) for model in polynomial_global_models],
            "polynomial_coefficients_uv": np.asarray(polynomial_global_poly["coefficients"], dtype=np.float64).tolist(),
            "terms_powers_xy": [[int(i), int(j)] for i, j in polynomial_global_poly["powers"]],
            "homography_fit_on_polynomial_centres": {
                "inlier_count": int(poly_centres_h_mask.sum()),
                "rmse": float(np.sqrt(np.mean(poly_centres_h_errors**2))),
                "median": float(np.median(poly_centres_h_errors)),
                "max": float(np.max(poly_centres_h_errors)),
            },
        },
        "comparison": {
            "baseline_H_local_plus_H_global": {
                "rmse_grid": float(np.sqrt(np.mean(source_h_global_errors**2))),
                "median_grid": float(np.median(source_h_global_errors)),
                "max_grid": float(np.max(source_h_global_errors)),
            },
            "H_local_plus_Polynomial_global": polynomial_report(source_global_poly),
            "Polynomial_local_plus_H_global": {
                "rmse_grid": float(np.sqrt(np.mean(poly_centres_h_errors**2))),
                "median_grid": float(np.median(poly_centres_h_errors)),
                "max_grid": float(np.max(poly_centres_h_errors)),
            },
            "Polynomial_local_plus_Polynomial_global": polynomial_report(polynomial_global_poly),
            "pose_quality": pose_quality,
        },
        "global_mapping": {
            "mapping_type": "polynomial_wide_pixel_to_ptz_lattice",
            "degree": int(polynomial_global_poly["degree"]),
            "terms_powers_xy": [[int(i), int(j)] for i, j in polynomial_global_poly["powers"]],
            "coefficients_uv": np.asarray(polynomial_global_poly["coefficients"], dtype=np.float64).tolist(),
            "input_normalization": {"width": wide_width, "height": wide_height},
            "state_surface": source_surface_model["state_surface"],
            "state_grid_actual_after_capture": state_grid.tolist(),
            "linearity_probe": source_surface_model["linearity_probe"],
            "pchip_trigger": source_surface_model.get("pchip_trigger"),
        },
    }
    write_json(output_dir / "polynomial_both_stages_mapping.json", result)
    write_json(output_dir / "global_model_comparison.json", result["comparison"])
    (output_dir / "README.md").write_text(
        "# Polynomial both-stages experiment\n\n"
        "Thử nghiệm này thay cả hai tầng: Polynomial bậc 2 cho từng cặp PTZ->Wide và Polynomial robust cho Wide->grid.\n"
        "Tầng grid->pan/tilt/zoom vẫn dùng pose surface hiện có. Không điều khiển camera.\n\n"
        f"- Local Polynomial degree: {args.local_degree}.\n"
        f"- Global Polynomial selected degree: {int(polynomial_global_poly['degree'])}.\n"
        "- `local_pair_comparison.csv`: so sánh tâm từ H cũ và Polynomial cho từng PTZ.\n"
        "- `polynomial_both_stages_mapping.json`: đầy đủ hệ số và chất lượng mô hình.\n"
        "- `diagnostics/local_h_centres_vs_polynomial_centres.jpg`: cam = tâm H, xanh = tâm Polynomial.\n"
        "- `diagnostics/both_stages_polynomial_grid_target_vs_prediction.jpg`: grid thật và dự đoán.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "global_degree": int(polynomial_global_poly["degree"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
