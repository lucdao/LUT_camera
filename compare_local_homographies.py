#!/usr/bin/env python3
"""Visualize how the 30 local 3x3 PTZ->Wide homographies differ by grid area.

This is a geometry-only diagnostic. It does not run feature extraction and does
not move a camera. For every selected PTZ/Wide pair it compares:

* normalized H coefficients;
* the projected PTZ image footprint on the Wide image;
* a canonical PTZ pixel grid projected to Wide;
* displacement from the central grid homography.
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


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def normalize_h(matrix: Any) -> np.ndarray:
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(h).all():
        raise RuntimeError(f"H không hợp lệ: shape={h.shape}")
    scale = h[2, 2]
    if abs(scale) < 1e-12:
        scale = np.linalg.norm(h)
    if abs(scale) < 1e-12:
        raise RuntimeError("Không chuẩn hóa được H vì scale gần 0.")
    return h / scale


def transform(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    output = cv2.perspectiveTransform(
        np.asarray(points, dtype=np.float32).reshape(-1, 1, 2),
        h.astype(np.float64),
    ).reshape(-1, 2).astype(np.float64)
    return output


def finite_points(points: np.ndarray) -> bool:
    return bool(np.isfinite(points).all())


def load_selected(mapping_dir: Path) -> tuple[Path, np.ndarray, list[dict[str, Any]], int, int]:
    selected_path = mapping_dir / "selected_homographies.json"
    if not selected_path.is_file():
        raise RuntimeError(f"Không tìm thấy {selected_path}")
    selected = json.loads(selected_path.read_text(encoding="utf-8")).get("selected", [])
    selected = sorted(selected, key=lambda item: (int(item["row"]), int(item["column"])))
    if len(selected) != 30:
        raise RuntimeError(f"Cần đúng 30 H local, hiện có {len(selected)}.")

    # selected_homographies stores image paths relative to the calibration
    # run root, and mapping_dir is .../<run_root>/02_... .
    run_root = mapping_dir.parent
    wide_path = run_root / selected[0]["wide_image"]
    wide_image = cv2.imread(str(wide_path), cv2.IMREAD_COLOR)
    if wide_image is None:
        raise RuntimeError(f"Không đọc được ảnh Wide: {wide_path}")
    ptz_path = run_root / selected[0]["ptz_image"]
    ptz_image = cv2.imread(str(ptz_path), cv2.IMREAD_COLOR)
    if ptz_image is None:
        raise RuntimeError(f"Không đọc được ảnh PTZ: {ptz_path}")
    wide_height, wide_width = wide_image.shape[:2]
    ptz_height, ptz_width = ptz_image.shape[:2]
    return wide_path, wide_image, selected, ptz_width, ptz_height


def canonical_grid(width: int, height: int, columns: int = 7, rows: int = 5) -> np.ndarray:
    # Include the image boundary: this makes the change in projected FOV visible.
    xs = np.linspace(0.0, width - 1.0, columns, dtype=np.float64)
    ys = np.linspace(0.0, height - 1.0, rows, dtype=np.float64)
    return np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)


def representative_indices(selected: list[dict[str, Any]]) -> list[int]:
    wanted = {
        (0, 0): "top-left",
        (0, 2): "top-centre",
        (0, 4): "top-right",
        (2, 0): "middle-left",
        (2, 2): "centre",
        (2, 4): "middle-right",
        (5, 0): "bottom-left",
        (5, 2): "bottom-centre",
        (5, 4): "bottom-right",
    }
    result: list[int] = []
    for index, item in enumerate(selected):
        if (int(item["row"]), int(item["column"])) in wanted:
            result.append(index)
    return result


def color_for_grid(row: int, column: int, rows: int = 6, columns: int = 5) -> tuple[int, int, int]:
    # A stable HSV palette: hue changes mostly with grid location.
    hue = int(179.0 * (0.65 * column / max(columns - 1, 1) + 0.35 * row / max(rows - 1, 1))) % 180
    hsv = np.uint8([[[hue, 220, 235]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw_polyline(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int], thickness: int = 2) -> None:
    if not finite_points(points):
        return
    for row in range(5):
        line = points[row * 7 : (row + 1) * 7]
        for first, second in zip(line[:-1], line[1:], strict=True):
            cv2.line(image, tuple(np.round(first).astype(int)), tuple(np.round(second).astype(int)), color, thickness, cv2.LINE_AA)
    for column in range(7):
        line = points[column::7]
        for first, second in zip(line[:-1], line[1:], strict=True):
            cv2.line(image, tuple(np.round(first).astype(int)), tuple(np.round(second).astype(int)), color, thickness, cv2.LINE_AA)


def scale_points(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float64).copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return scaled


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255), scale: float = 0.55) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_all_footprints(
    wide_image: np.ndarray,
    selected: list[dict[str, Any]],
    projected: list[np.ndarray],
    output_path: Path,
) -> None:
    image = wide_image.copy()
    for item, points in zip(selected, projected, strict=True):
        height, width = image.shape[:2]
        sx = sy = 1.0
        footprint = points[[0, 6, 34, 28]]
        footprint_int = np.round(footprint).astype(int).reshape(-1, 1, 2)
        color = color_for_grid(int(item["row"]), int(item["column"]))
        cv2.polylines(image, [footprint_int], True, color, 2, cv2.LINE_AA)
        centre = np.mean(footprint, axis=0)
        cv2.circle(image, tuple(np.round(centre).astype(int)), 6, color, -1, cv2.LINE_AA)
        label_xy = tuple(np.round(centre).astype(int) + np.asarray([7, -7]))
        draw_text(image, f"{int(item['ptz_index']):02d}", label_xy, color, 0.42)
    draw_text(image, "30 local H footprints on Wide | color = PTZ grid location", (25, 35), (255, 255, 255), 0.72)
    draw_text(image, "Each polygon is the same PTZ image transformed by its own H_ptz_to_wide", (25, 68), (255, 255, 255), 0.52)
    cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 93])


def make_representative_contact_sheet(
    wide_image: np.ndarray,
    selected: list[dict[str, Any]],
    projected: list[np.ndarray],
    output_path: Path,
) -> None:
    target_width = 640
    target_height = int(round(wide_image.shape[0] * target_width / wide_image.shape[1]))
    panels: list[np.ndarray] = []
    for index in representative_indices(selected):
        panel = cv2.resize(wide_image, (target_width, target_height), interpolation=cv2.INTER_AREA)
        points = scale_points(projected[index], target_width / wide_image.shape[1], target_height / wide_image.shape[0])
        draw_polyline(panel, points, (0, 220, 0), 2)
        corners = points[[0, 6, 34, 28]]
        cv2.polylines(panel, [np.round(corners).astype(int).reshape(-1, 1, 2)], True, (0, 0, 255), 3, cv2.LINE_AA)
        center = np.mean(corners, axis=0)
        cv2.drawMarker(panel, tuple(np.round(center).astype(int)), (255, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        item = selected[index]
        title = f"row={int(item['row'])} col={int(item['column'])} | PTZ {int(item['ptz_index']):02d}"
        draw_text(panel, title, (15, 28), (255, 255, 255), 0.58)
        panels.append(panel)
    sheet = np.zeros((target_height * 3, target_width * 3, 3), dtype=np.uint8)
    for index, panel in enumerate(panels):
        row, column = divmod(index, 3)
        sheet[row * target_height : (row + 1) * target_height, column * target_width : (column + 1) * target_width] = panel
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 93])


def make_value_heatmap(values: np.ndarray, labels: list[str], output_path: Path, rows: int = 6, columns: int = 5) -> None:
    cell_width, cell_height = 180, 88
    panel_width, panel_height = columns * cell_width, (rows + 1) * cell_height
    canvas = np.zeros((2 * panel_height, 4 * panel_width, 3), dtype=np.uint8)
    for coefficient_index, label in enumerate(labels):
        panel_row, panel_column = divmod(coefficient_index, 4)
        x0, y0 = panel_column * panel_width, panel_row * panel_height
        panel = np.full((panel_height, panel_width, 3), 35, dtype=np.uint8)
        data = values[coefficient_index]
        finite = data[np.isfinite(data)]
        low, high = float(np.min(finite)), float(np.max(finite))
        scale = max(high - low, 1e-12)
        draw_text(panel, f"{label}: min={low:.4g} max={high:.4g}", (8, 22), (255, 255, 255), 0.48)
        for row in range(rows):
            for column in range(columns):
                value = float(data[row, column])
                normalized = int(np.clip(round(255.0 * (value - low) / scale), 0, 255))
                color = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[normalized]]), cv2.COLORMAP_TURBO)[0, 0])
                left = column * cell_width
                top = (row + 1) * cell_height
                cv2.rectangle(panel, (left, top), (left + cell_width - 1, top + cell_height - 1), color, -1)
                cv2.rectangle(panel, (left, top), (left + cell_width - 1, top + cell_height - 1), (20, 20, 20), 1)
                text = f"{value:.3g}"
                size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)[0]
                cv2.putText(panel, text, (left + (cell_width - size[0]) // 2, top + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(panel, text, (left + (cell_width - size[0]) // 2, top + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
        canvas[y0 : y0 + panel_height, x0 : x0 + panel_width] = panel
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def make_grid_heatmap(values: np.ndarray, title: str, output_path: Path, rows: int = 6, columns: int = 5) -> None:
    cell_width, cell_height = 250, 105
    canvas = np.full(((rows + 1) * cell_height, columns * cell_width, 3), 40, dtype=np.uint8)
    finite = values[np.isfinite(values)]
    low, high = float(np.min(finite)), float(np.max(finite))
    scale = max(high - low, 1e-12)
    draw_text(canvas, f"{title} | min={low:.4g}, max={high:.4g}", (10, 30), (255, 255, 255), 0.68)
    for row in range(rows):
        for column in range(columns):
            value = float(values[row, column])
            normalized = int(np.clip(round(255.0 * (value - low) / scale), 0, 255))
            color = tuple(int(v) for v in cv2.applyColorMap(np.uint8([[normalized]]), cv2.COLORMAP_TURBO)[0, 0])
            left, top = column * cell_width, (row + 1) * cell_height
            cv2.rectangle(canvas, (left, top), (left + cell_width - 1, top + cell_height - 1), color, -1)
            cv2.rectangle(canvas, (left, top), (left + cell_width - 1, top + cell_height - 1), (20, 20, 20), 2)
            text = f"r{row} c{column}\n{value:.3f}"
            cv2.putText(canvas, f"r{row} c{column}", (left + 12, top + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, f"{value:.3f}", (left + 12, top + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-dir", type=Path, required=True, help="Directory containing selected_homographies.json.")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    mapping_dir = args.mapping_dir.resolve()
    wide_path, wide_image, selected, ptz_width, ptz_height = load_selected(mapping_dir)
    output_dir = (args.output_dir or mapping_dir / f"h_matrix_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    canonical = canonical_grid(ptz_width, ptz_height)
    matrices = [normalize_h(item["H_ptz_to_wide"]) for item in selected]
    projected = [transform(canonical, h) for h in matrices]
    centre_index = min(range(len(selected)), key=lambda index: (abs(int(selected[index]["row"]) - 2.5) + abs(int(selected[index]["column"]) - 2)))
    reference_h = matrices[centre_index]
    reference_projected = projected[centre_index]

    rows, columns = 6, 5
    coefficient_names = ["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21"]
    coefficient_values = np.full((8, rows, columns), np.nan, dtype=np.float64)
    displacement_values = np.full((rows, columns), np.nan, dtype=np.float64)
    corner_displacement_values = np.full((rows, columns), np.nan, dtype=np.float64)
    shape_displacement_values = np.full((rows, columns), np.nan, dtype=np.float64)
    report_rows: list[dict[str, Any]] = []
    for index, (item, h, points) in enumerate(zip(selected, matrices, projected, strict=True)):
        row, column = int(item["row"]), int(item["column"])
        coefficient_values[:, row, column] = h.reshape(-1)[:8]
        displacement = np.linalg.norm(points - reference_projected, axis=1)
        corners = points[[0, 6, 34, 28]]
        reference_corners = reference_projected[[0, 6, 34, 28]]
        corner_displacement = np.linalg.norm(corners - reference_corners, axis=1)
        # Remove the translation of each footprint. This isolates changes in
        # scale, rotation, skew and projective distortion from mere Wide
        # coverage movement.
        local_center = points[2 * 7 + 3]
        reference_center = reference_projected[2 * 7 + 3]
        shape_displacement = np.linalg.norm(
            (points - local_center) - (reference_projected - reference_center),
            axis=1,
        )
        displacement_values[row, column] = float(np.mean(displacement))
        corner_displacement_values[row, column] = float(np.mean(corner_displacement))
        shape_displacement_values[row, column] = float(np.mean(shape_displacement))
        report_rows.append(
            {
                "ptz_index": int(item["ptz_index"]),
                "row": row,
                "column": column,
                "wide_center_x": float(item["mapped_ptz_center_on_wide"][0]),
                "wide_center_y": float(item["mapped_ptz_center_on_wide"][1]),
                "h00": float(h[0, 0]),
                "h01": float(h[0, 1]),
                "h02": float(h[0, 2]),
                "h10": float(h[1, 0]),
                "h11": float(h[1, 1]),
                "h12": float(h[1, 2]),
                "h20": float(h[2, 0]),
                "h21": float(h[2, 1]),
                "h22": float(h[2, 2]),
                "frobenius_delta_vs_centre": float(np.linalg.norm(h - reference_h)),
                "mean_projected_grid_displacement_vs_centre_px": float(np.mean(displacement)),
                "median_projected_grid_displacement_vs_centre_px": float(np.median(displacement)),
                "max_projected_grid_displacement_vs_centre_px": float(np.max(displacement)),
                "mean_corner_displacement_vs_centre_px": float(np.mean(corner_displacement)),
                "max_corner_displacement_vs_centre_px": float(np.max(corner_displacement)),
                "mean_shape_displacement_after_translation_alignment_px": float(np.mean(shape_displacement)),
                "max_shape_displacement_after_translation_alignment_px": float(np.max(shape_displacement)),
                "local_inliers": int(item.get("inliers", item.get("saved_h_inlier_count", 0))),
                "local_median_reprojection_error_px": item.get("median_reprojection_error_px"),
            }
        )

    make_all_footprints(wide_image, selected, projected, output_dir / "h_all_footprints_on_wide.jpg")
    make_representative_contact_sheet(wide_image, selected, projected, output_dir / "h_9_regions_projected_grid_contact_sheet.jpg")
    make_value_heatmap(coefficient_values, coefficient_names, output_dir / "h_normalized_coefficients_heatmap.jpg")
    make_grid_heatmap(displacement_values, "Mean projected PTZ-grid displacement vs central H (pixels)", output_dir / "h_mean_displacement_vs_centre_heatmap.jpg")
    make_grid_heatmap(corner_displacement_values, "Mean projected corner displacement vs central H (pixels)", output_dir / "h_corner_displacement_vs_centre_heatmap.jpg")
    make_grid_heatmap(shape_displacement_values, "Mean H shape difference after translation alignment (pixels)", output_dir / "h_shape_difference_vs_centre_heatmap.jpg")

    csv_path = output_dir / "h_matrix_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report_rows[0]))
        writer.writeheader()
        writer.writerows(report_rows)
    write_json(
        output_dir / "h_matrix_comparison.json",
        {
            "status": "complete",
            "source_selected_homographies": str(mapping_dir / "selected_homographies.json"),
            "wide_reference": str(wide_path),
            "count": len(selected),
            "reference": {
                "ptz_index": int(selected[centre_index]["ptz_index"]),
                "row": int(selected[centre_index]["row"]),
                "column": int(selected[centre_index]["column"]),
                "normalized_H_ptz_to_wide": reference_h.tolist(),
            },
            "canonical_ptz_grid": {"width": ptz_width, "height": ptz_height, "columns": 7, "rows": 5},
            "interpretation": {
                "matrix_coefficients_are_normalized_by_h22": True,
                "visual_displacement_is_more_meaningful_than_raw_H_coefficient_difference": True,
                "positive_mean_projected_grid_displacement_means_the_local_H_maps_the_same_PTz_pixel_grid_differently_from_the_centre_H": True,
            },
            "rows": report_rows,
        },
    )
    (output_dir / "README.md").write_text(
        "# Local H matrix comparison\n\n"
        "Báo cáo so sánh 30 ma trận H_ptz_to_wide 3x3. H được chuẩn hóa về H[2,2]=1.\n\n"
        f"Ma trận tham chiếu là PTZ index {int(selected[centre_index]['ptz_index'])}, row={int(selected[centre_index]['row'])}, col={int(selected[centre_index]['column'])}.\n\n"
        "- `h_all_footprints_on_wide.jpg`: 30 footprint PTZ sau khi chiếu lên Wide.\n"
        "- `h_9_regions_projected_grid_contact_sheet.jpg`: góc trái/phải, giữa các cạnh và trung tâm. Lưới xanh là cùng một lưới pixel PTZ được chiếu bởi H tương ứng; viền đỏ là footprint.\n"
        "- `h_normalized_coefficients_heatmap.jpg`: hệ số H đã chuẩn hóa theo từng vị trí grid.\n"
        "- `h_mean_displacement_vs_centre_heatmap.jpg`: sai khác trung bình bằng pixel so với H trung tâm.\n"
        "- `h_corner_displacement_vs_centre_heatmap.jpg`: sai khác footprint ở bốn góc PTZ so với H trung tâm.\n"
        "- `h_shape_difference_vs_centre_heatmap.jpg`: sai khác hình dạng sau khi đã loại bỏ dịch chuyển vị trí; phản ánh scale/rotation/skew/projective distortion.\n"
        "- `h_matrix_comparison.csv/json`: số liệu đầy đủ cho từng cặp.\n\n"
        "Không nên kết luận chỉ từ độ lệch hệ số H; sai khác footprint và sai khác các điểm lưới PTZ trên Wide có ý nghĩa hình học trực quan hơn.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir), "reference_ptz_index": int(selected[centre_index]["ptz_index"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
