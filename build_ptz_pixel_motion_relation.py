#!/usr/bin/env python3
"""Build a PTZ pixel-error -> pan/tilt-motion relation from saved local H.

The 30 saved homographies are not interpolated as a function of pose here.
Instead, every ordered pair of PTZ poses is linked through the common Wide
coordinate frame.  For a point that is at the centre of destination pose j,
we project that point back into source pose i.  This creates a teacher sample:

    centre_i - pixel_i  ->  (pan_j - pan_i, tilt_j - tilt_i)

The script is offline-only.  It does not connect to or move a camera.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_run_path(run_root: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return run_root / candidate


def image_size(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được ảnh để lấy kích thước: {path}")
    height, width = image.shape[:2]
    return width, height


def normalize_h(matrix: Any) -> np.ndarray:
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"H phải là 3x3, nhận được {h.shape}")
    scale = h[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(h)
    if abs(scale) < 1e-12:
        raise ValueError("H suy biến khi chuẩn hóa")
    return h / scale


def project(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    result = cv2.perspectiveTransform(points.astype(np.float64), h.astype(np.float64))
    return result.reshape(-1, 2)


def in_bounds(points: np.ndarray, width: int, height: int, margin: float) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= margin)
        & (x < width - margin)
        & (y >= margin)
        & (y < height - margin)
    )


def design_matrix(errors_px: np.ndarray, width: int, height: int, degree: int) -> tuple[np.ndarray, list[str]]:
    normalized = np.asarray(errors_px, dtype=np.float64).copy()
    normalized[:, 0] /= max(width / 2.0, 1.0)
    normalized[:, 1] /= max(height / 2.0, 1.0)
    ex = normalized[:, 0]
    ey = normalized[:, 1]
    columns = [ex, ey]
    names = ["ex_norm", "ey_norm"]
    if degree >= 2:
        columns.extend([ex * ex, ex * ey, ey * ey])
        names.extend(["ex2_norm", "exey_norm", "ey2_norm"])
    return np.column_stack(columns), names


def fit_model(errors_px: np.ndarray, deltas: np.ndarray, width: int, height: int, degree: int) -> dict[str, Any]:
    x, names = design_matrix(errors_px, width, height, degree)
    coefficients, _, rank, singular_values = np.linalg.lstsq(x, deltas, rcond=None)
    predicted = x @ coefficients
    residuals = predicted - deltas
    norms = np.linalg.norm(residuals, axis=1)
    return {
        "degree": degree,
        "feature_names": names,
        "coefficients": coefficients.tolist(),
        "rank": int(rank),
        "singular_values": singular_values.tolist(),
        "sample_count": int(len(errors_px)),
        "rmse_pan": float(np.sqrt(np.mean(residuals[:, 0] ** 2))),
        "rmse_tilt": float(np.sqrt(np.mean(residuals[:, 1] ** 2))),
        "rmse_vector": float(np.sqrt(np.mean(norms ** 2))),
        "median_vector_error": float(np.median(norms)),
        "p95_vector_error": float(np.percentile(norms, 95)),
    }


def predict_model(model: dict[str, Any], errors_px: np.ndarray, width: int, height: int) -> np.ndarray:
    x, _ = design_matrix(errors_px, width, height, int(model["degree"]))
    return x @ np.asarray(model["coefficients"], dtype=np.float64)


def leave_one_source_out(
    errors_px: np.ndarray,
    deltas: np.ndarray,
    source_indices: np.ndarray,
    width: int,
    height: int,
    degree: int,
) -> dict[str, Any]:
    residuals: list[np.ndarray] = []
    group_reports: list[dict[str, Any]] = []
    for source in sorted(set(int(x) for x in source_indices)):
        train = source_indices != source
        test = ~train
        if int(np.count_nonzero(train)) < 6 or not np.any(test):
            continue
        model = fit_model(errors_px[train], deltas[train], width, height, degree)
        prediction = predict_model(model, errors_px[test], width, height)
        error = prediction - deltas[test]
        residuals.append(error)
        vector_norm = np.linalg.norm(error, axis=1)
        group_reports.append({
            "source_ptz_index": source,
            "test_count": int(np.count_nonzero(test)),
            "rmse_vector": float(np.sqrt(np.mean(vector_norm ** 2))),
            "median_vector_error": float(np.median(vector_norm)),
            "p95_vector_error": float(np.percentile(vector_norm, 95)),
        })
    if not residuals:
        return {"group_count": 0, "groups": []}
    all_residuals = np.vstack(residuals)
    norms = np.linalg.norm(all_residuals, axis=1)
    return {
        "group_count": len(group_reports),
        "groups": group_reports,
        "rmse_pan": float(np.sqrt(np.mean(all_residuals[:, 0] ** 2))),
        "rmse_tilt": float(np.sqrt(np.mean(all_residuals[:, 1] ** 2))),
        "rmse_vector": float(np.sqrt(np.mean(norms ** 2))),
        "median_vector_error": float(np.median(norms)),
        "p95_vector_error": float(np.percentile(norms, 95)),
    }


def load_entries(mapping_dir: Path) -> tuple[list[dict[str, Any]], Path, tuple[int, int], tuple[int, int]]:
    source_path = mapping_dir / "selected_homographies.json"
    payload = read_json(source_path)
    run_root = mapping_dir.parent
    entries: list[dict[str, Any]] = []
    for raw in payload["selected"]:
        ptz_path = resolve_run_path(run_root, raw.get("ptz_image") or raw["raw_image"])
        wide_path = resolve_run_path(run_root, raw["wide_image"])
        entries.append({
            "ptz_index": int(raw["ptz_index"]),
            "row": int(raw["row"]),
            "column": int(raw["column"]),
            "pan": float(raw["actual_after_capture"]["pan"]),
            "tilt": float(raw["actual_after_capture"]["tilt"]),
            "zoom": float(raw["actual_after_capture"].get("zoom", 0.0)),
            "H": normalize_h(raw["H_ptz_to_wide"]),
            "inliers": int(raw.get("inliers", 0)),
            "median_reprojection_error_px": float(raw.get("median_reprojection_error_px", float("nan"))),
            "center_inside_wide": bool(raw.get("center_inside_wide", False)),
            "ptz_path": str(ptz_path),
            "wide_path": str(wide_path),
        })
    if not entries:
        raise RuntimeError("Không có selected homographies")
    ptz_width, ptz_height = image_size(Path(entries[0]["ptz_path"]))
    wide_width, wide_height = image_size(Path(entries[0]["wide_path"]))
    return entries, source_path, (ptz_width, ptz_height), (wide_width, wide_height)


def build_center_goal_samples(
    entries: list[dict[str, Any]],
    ptz_size: tuple[int, int],
    wide_size: tuple[int, int],
    margin_px: float,
) -> list[dict[str, Any]]:
    ptz_width, ptz_height = ptz_size
    wide_width, wide_height = wide_size
    centre = np.array([[ptz_width / 2.0, ptz_height / 2.0]], dtype=np.float64)
    samples: list[dict[str, Any]] = []
    for source in entries:
        h_source_inv = np.linalg.inv(source["H"])
        for destination in entries:
            if source["ptz_index"] == destination["ptz_index"]:
                continue
            destination_wide = project(destination["H"], centre)[0]
            if not bool(in_bounds(destination_wide.reshape(1, 2), wide_width, wide_height, margin_px)[0]):
                continue
            source_pixel = project(h_source_inv, destination_wide.reshape(1, 2))[0]
            if not bool(in_bounds(source_pixel.reshape(1, 2), ptz_width, ptz_height, margin_px)[0]):
                continue
            error_to_centre = centre[0] - source_pixel
            samples.append({
                "source_ptz_index": source["ptz_index"],
                "destination_ptz_index": destination["ptz_index"],
                "source_row": source["row"],
                "source_column": source["column"],
                "destination_row": destination["row"],
                "destination_column": destination["column"],
                "source_pan": source["pan"],
                "source_tilt": source["tilt"],
                "destination_pan": destination["pan"],
                "destination_tilt": destination["tilt"],
                "zoom": source["zoom"],
                "source_pixel_u": float(source_pixel[0]),
                "source_pixel_v": float(source_pixel[1]),
                "source_centre_u": float(centre[0, 0]),
                "source_centre_v": float(centre[0, 1]),
                "error_to_centre_u": float(error_to_centre[0]),
                "error_to_centre_v": float(error_to_centre[1]),
                "destination_centre_wide_u": float(destination_wide[0]),
                "destination_centre_wide_v": float(destination_wide[1]),
                "delta_pan": float(destination["pan"] - source["pan"]),
                "delta_tilt": float(destination["tilt"] - source["tilt"]),
            })
    return samples


def build_pair_transfer_samples(
    entries: list[dict[str, Any]],
    ptz_size: tuple[int, int],
    wide_size: tuple[int, int],
    margin_px: float,
    grid_x: int = 9,
    grid_y: int = 7,
) -> list[dict[str, Any]]:
    """Create arbitrary-point transfers through the common Wide frame."""
    ptz_width, ptz_height = ptz_size
    wide_width, wide_height = wide_size
    xs = np.linspace(margin_px, ptz_width - margin_px, grid_x)
    ys = np.linspace(margin_px, ptz_height - margin_px, grid_y)
    source_grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)
    samples: list[dict[str, Any]] = []
    for source in entries:
        for destination in entries:
            if source["ptz_index"] == destination["ptz_index"]:
                continue
            wide_points = project(source["H"], source_grid)
            source_valid = in_bounds(source_grid, ptz_width, ptz_height, margin_px)
            wide_valid = in_bounds(wide_points, wide_width, wide_height, margin_px)
            if not np.any(source_valid & wide_valid):
                continue
            destination_points = project(np.linalg.inv(destination["H"]), wide_points)
            destination_valid = in_bounds(destination_points, ptz_width, ptz_height, margin_px)
            valid = source_valid & wide_valid & destination_valid
            for source_point, wide_point, destination_point in zip(source_grid[valid], wide_points[valid], destination_points[valid]):
                pixel_delta = destination_point - source_point
                source_error = np.array([ptz_width / 2.0, ptz_height / 2.0]) - source_point
                destination_error = np.array([ptz_width / 2.0, ptz_height / 2.0]) - destination_point
                samples.append({
                    "source_ptz_index": source["ptz_index"],
                    "destination_ptz_index": destination["ptz_index"],
                    "source_pan": source["pan"],
                    "source_tilt": source["tilt"],
                    "destination_pan": destination["pan"],
                    "destination_tilt": destination["tilt"],
                    "zoom": source["zoom"],
                    "source_pixel_u": float(source_point[0]),
                    "source_pixel_v": float(source_point[1]),
                    "wide_pixel_u": float(wide_point[0]),
                    "wide_pixel_v": float(wide_point[1]),
                    "destination_pixel_u": float(destination_point[0]),
                    "destination_pixel_v": float(destination_point[1]),
                    "pixel_delta_u": float(pixel_delta[0]),
                    "pixel_delta_v": float(pixel_delta[1]),
                    "source_error_to_centre_u": float(source_error[0]),
                    "source_error_to_centre_v": float(source_error[1]),
                    "destination_error_to_centre_u": float(destination_error[0]),
                    "destination_error_to_centre_v": float(destination_error[1]),
                    "delta_pan": float(destination["pan"] - source["pan"]),
                    "delta_tilt": float(destination["tilt"] - source["tilt"]),
                })
    return samples


def write_samples_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    if not samples:
        return
    fields = list(samples[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--margin-px", type=float, default=20.0)
    args = parser.parse_args()

    entries, source_path, ptz_size, wide_size = load_entries(args.mapping_dir)
    samples = build_center_goal_samples(entries, ptz_size, wide_size, args.margin_px)
    pair_samples = build_pair_transfer_samples(entries, ptz_size, wide_size, args.margin_px)
    if len(samples) < 10:
        raise RuntimeError(f"Chỉ tạo được {len(samples)} mẫu, không đủ để fit mô hình")

    errors = np.array([[x["error_to_centre_u"], x["error_to_centre_v"]] for x in samples], dtype=np.float64)
    deltas = np.array([[x["delta_pan"], x["delta_tilt"]] for x in samples], dtype=np.float64)
    source_indices = np.array([x["source_ptz_index"] for x in samples], dtype=np.int64)
    models: dict[str, Any] = {}
    for degree, name in ((1, "linear_no_intercept"), (2, "quadratic_no_intercept")):
        model = fit_model(errors, deltas, ptz_size[0], ptz_size[1], degree)
        model["leave_one_source_out"] = leave_one_source_out(errors, deltas, source_indices, ptz_size[0], ptz_size[1], degree)
        models[name] = model

    args.output_dir.mkdir(parents=True, exist_ok=True)
    serial_entries = []
    for item in entries:
        serial = {key: value for key, value in item.items() if key != "H"}
        serial["H_ptz_to_wide"] = item["H"].tolist()
        serial_entries.append(serial)
    write_json(args.output_dir / "ptz_motion_relation_samples.json", {
        "source_selected_homographies": str(source_path),
        "ptz_image_size": {"width": ptz_size[0], "height": ptz_size[1]},
        "wide_image_size": {"width": wide_size[0], "height": wide_size[1]},
        "margin_px": args.margin_px,
        "sample_definition": "destination PTZ centre projected back through H_destination^-1 H_source",
        "sample_count": len(samples),
        "samples": samples,
    })
    write_samples_csv(args.output_dir / "ptz_motion_relation_samples.csv", samples)
    write_json(args.output_dir / "ptz_pair_transfer_samples.json", {
        "source_selected_homographies": str(source_path),
        "ptz_image_size": {"width": ptz_size[0], "height": ptz_size[1]},
        "wide_image_size": {"width": wide_size[0], "height": wide_size[1]},
        "margin_px": args.margin_px,
        "sample_definition": "p_destination = H_destination^-1 H_source p_source",
        "sample_count": len(pair_samples),
        "samples": pair_samples,
    })
    write_samples_csv(args.output_dir / "ptz_pair_transfer_samples.csv", pair_samples)
    write_json(args.output_dir / "ptz_motion_relation_models.json", {
        "source_selected_homographies": str(source_path),
        "ptz_image_size": {"width": ptz_size[0], "height": ptz_size[1]},
        "wide_image_size": {"width": wide_size[0], "height": wide_size[1]},
        "zoom_values": sorted({x["zoom"] for x in entries}),
        "sample_count": len(samples),
        "pair_transfer_sample_count": len(pair_samples),
        "models": models,
        "entries": serial_entries,
    })
    print(json.dumps({
        "status": "complete",
        "sample_count": len(samples),
        "pair_transfer_sample_count": len(pair_samples),
        "ptz_image_size": ptz_size,
        "wide_image_size": wide_size,
        "models": {
            name: {
                "rmse_vector": value["rmse_vector"],
                "median_vector_error": value["median_vector_error"],
                "p95_vector_error": value["p95_vector_error"],
                "loo_rmse_vector": value["leave_one_source_out"].get("rmse_vector"),
                "loo_median_vector_error": value["leave_one_source_out"].get("median_vector_error"),
            }
            for name, value in models.items()
        },
        "output_dir": str(args.output_dir),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
