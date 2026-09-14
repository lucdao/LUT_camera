#!/usr/bin/env python3
"""Create visual checks for the 30 PTZ-centre-to-Wide correspondences."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def load_selected(run_root: Path) -> list[dict[str, Any]]:
    path = run_root / "02_aliked_lightglue_wide_ptz_mapping" / "selected_homographies.json"
    if not path.is_file():
        raise RuntimeError(f"Không tìm thấy selected_homographies.json: {path}")
    selected = json.loads(path.read_text(encoding="utf-8")).get("selected", [])
    selected = sorted(selected, key=lambda item: int(item["ptz_index"]))
    if len(selected) != 30:
        raise RuntimeError(f"Cần đúng 30 cặp homography, hiện có {len(selected)}.")
    return selected


def load_reference_wide(run_root: Path) -> tuple[Path, str]:
    metadata_path = run_root / "00_wide_current_and_new_set" / "wide_capture_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    item = metadata.get("current_reference")
    if not item or item.get("status") != "ok":
        raise RuntimeError("Không tìm thấy current_reference Wide hợp lệ.")
    return run_root / item["image"], "current_reference"


def draw_text(image: np.ndarray, text: str, xy: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def add_margin(image: np.ndarray, points: list[tuple[float, float]], margin: int = 220) -> tuple[np.ndarray, int]:
    del points  # A fixed margin keeps all outputs comparable.
    return cv2.copyMakeBorder(image, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=(35, 35, 35)), margin


def draw_wide_centres(
    image_path: Path,
    output_path: Path,
    selected: list[dict[str, Any]],
    title: str,
) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được Wide image: {image_path}")
    canvas, offset = add_margin(image, [])
    for item in selected:
        actual = np.asarray(item["mapped_ptz_center_on_wide"], dtype=np.float64)
        expected = np.asarray(item.get("expected_wide_anchor", [float("nan"), float("nan")]), dtype=np.float64)
        actual_xy = tuple(np.round(actual).astype(int) + offset)
        expected_xy = tuple(np.round(expected).astype(int) + offset) if np.isfinite(expected).all() else None
        cv2.circle(canvas, actual_xy, 10, (0, 220, 0), 3, cv2.LINE_AA)
        cv2.drawMarker(canvas, actual_xy, (0, 220, 0), cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
        if expected_xy is not None:
            cv2.drawMarker(canvas, expected_xy, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 24, 2, cv2.LINE_AA)
            cv2.line(canvas, expected_xy, actual_xy, (0, 255, 255), 1, cv2.LINE_AA)
        label = f"PTZ {int(item['ptz_index']):02d}"
        draw_text(canvas, label, (actual_xy[0] + 12, actual_xy[1] - 12), (0, 255, 0))
    draw_text(canvas, title, (20, 35), (255, 255, 255))
    draw_text(canvas, "GREEN = mapped PTZ centre | ORANGE = acquisition anchor | YELLOW = difference", (20, 65), (255, 255, 255))
    if not cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Không ghi được ảnh: {output_path}")


def draw_native_wide_pair(image_path: Path, output_path: Path, item: dict[str, Any]) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được Wide image: {image_path}")
    actual = np.asarray(item["mapped_ptz_center_on_wide"], dtype=np.float64)
    expected = np.asarray(item.get("expected_wide_anchor", [float("nan"), float("nan")]), dtype=np.float64)
    canvas, offset = add_margin(image, [])
    actual_xy = tuple(np.round(actual).astype(int) + offset)
    cv2.circle(canvas, actual_xy, 11, (0, 220, 0), 3, cv2.LINE_AA)
    cv2.drawMarker(canvas, actual_xy, (0, 220, 0), cv2.MARKER_CROSS, 32, 2, cv2.LINE_AA)
    if np.isfinite(expected).all():
        expected_xy = tuple(np.round(expected).astype(int) + offset)
        cv2.drawMarker(canvas, expected_xy, (0, 165, 255), cv2.MARKER_TILTED_CROSS, 26, 2, cv2.LINE_AA)
        cv2.line(canvas, expected_xy, actual_xy, (0, 255, 255), 1, cv2.LINE_AA)
    state = item.get("actual_after_capture", {})
    text = (
        f"PTZ {int(item['ptz_index']):02d} | row={int(item['row'])} col={int(item['column'])} | "
        f"pan={float(state.get('pan', 0.0)):.4f} tilt={float(state.get('tilt', 0.0)):.4f} zoom={float(state.get('zoom', 0.0)):.4f}"
    )
    draw_text(canvas, text, (20, 35), (255, 255, 255))
    if not cv2.imwrite(str(output_path), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Không ghi được ảnh: {output_path}")


def draw_ptz_centre(image_path: Path, output_path: Path, item: dict[str, Any]) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được PTZ image: {image_path}")
    height, width = image.shape[:2]
    centre = (width // 2, height // 2)
    cv2.drawMarker(image, centre, (0, 0, 255), cv2.MARKER_CROSS, max(40, min(width, height) // 8), 5, cv2.LINE_AA)
    cv2.circle(image, centre, max(18, min(width, height) // 32), (0, 0, 255), 3, cv2.LINE_AA)
    state = item.get("actual_after_capture", {})
    draw_text(
        image,
        f"PTZ {int(item['ptz_index']):02d} | row={int(item['row'])} col={int(item['column'])} | "
        f"pan={float(state.get('pan', 0.0)):.4f} tilt={float(state.get('tilt', 0.0)):.4f} zoom={float(state.get('zoom', 0.0)):.4f}",
        (30, 45),
        (0, 0, 255),
    )
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]):
        raise RuntimeError(f"Không ghi được ảnh: {output_path}")
    return image


def make_contact_sheet(images: list[np.ndarray], output_path: Path, columns: int, cell_width: int = 480) -> None:
    if not images:
        raise RuntimeError("Không có ảnh để tạo contact sheet.")
    rows = (len(images) + columns - 1) // columns
    cell_height = int(round(images[0].shape[0] * cell_width / images[0].shape[1]))
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        thumb = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width] = thumb
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True, help="Completed run containing selected_homographies.json.")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    selected = load_selected(run_root)
    reference_wide, reference_id = load_reference_wide(run_root)
    output_dir = (args.output_dir or run_root / f"03_center_visual_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ptz_dir = output_dir / "ptz_centres"
    native_wide_dir = output_dir / "wide_native_by_pair"
    ptz_dir.mkdir()
    native_wide_dir.mkdir()

    draw_wide_centres(
        reference_wide,
        output_dir / "wide_reference_30_ptz_centres.jpg",
        selected,
        f"30 PTZ centres projected on Wide reference ({reference_id})",
    )
    contact_images: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for item in selected:
        index = int(item["ptz_index"])
        ptz_path = run_root / item["ptz_image"]
        wide_path = run_root / item["wide_image"]
        draw_native_wide_pair(wide_path, native_wide_dir / f"ptz_{index:02d}_on_{item['wide_id']}.jpg", item)
        contact_images.append(draw_ptz_centre(ptz_path, ptz_dir / f"ptz_{index:02d}_centre.jpg", item))
        state = item.get("actual_after_capture", {})
        centre = item["mapped_ptz_center_on_wide"]
        rows.append(
            {
                "ptz_index": index,
                "row": int(item["row"]),
                "column": int(item["column"]),
                "wide_id": item["wide_id"],
                "wide_image": item["wide_image"],
                "ptz_image": item["ptz_image"],
                "mapped_wide_x": float(centre[0]),
                "mapped_wide_y": float(centre[1]),
                "expected_anchor_x": float(item["expected_wide_anchor"][0]),
                "expected_anchor_y": float(item["expected_wide_anchor"][1]),
                "center_prior_error_px": item.get("center_prior_error_px"),
                "pan": float(state["pan"]),
                "tilt": float(state["tilt"]),
                "zoom": float(state["zoom"]),
                "inliers": int(item["inliers"]),
                "inlier_ratio": float(item["inlier_ratio"]),
            }
        )
    make_contact_sheet(contact_images, output_dir / "ptz_30_centres_contact_sheet.jpg", columns=5)
    with (output_dir / "center_pairs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output_dir / "visual_check_metadata.json",
        {
            "status": "complete",
            "run_root": str(run_root),
            "source_selected_homographies": str(run_root / "02_aliked_lightglue_wide_ptz_mapping" / "selected_homographies.json"),
            "count": len(selected),
            "wide_reference": str(reference_wide),
            "outputs": {
                "wide_reference_overlay": "wide_reference_30_ptz_centres.jpg",
                "wide_native_by_pair": "wide_native_by_pair",
                "ptz_centres": "ptz_centres",
                "ptz_contact_sheet": "ptz_30_centres_contact_sheet.jpg",
                "pairs_csv": "center_pairs.csv",
            },
            "legend": {
                "green": "mapped PTZ image centre projected onto Wide",
                "orange": "saved acquisition Wide anchor used as a validation prior",
                "yellow": "vector from anchor to mapped centre",
                "red": "geometric centre of each PTZ image",
            },
        },
    )
    (output_dir / "README.md").write_text(
        "# Kiểm tra trực quan 30 tâm PTZ\n\n"
        "- `wide_reference_30_ptz_centres.jpg`: 30 tâm PTZ chiếu lên cùng Wide reference.\n"
        "- `wide_native_by_pair/`: mỗi cặp được vẽ trên đúng Wide image dùng để fit homography của cặp đó.\n"
        "- `ptz_centres/`: 30 ảnh PTZ raw, mỗi ảnh có dấu đỏ tại tâm hình học.\n"
        "- `ptz_30_centres_contact_sheet.jpg`: xem nhanh 30 ảnh PTZ.\n"
        "- `center_pairs.csv`: tọa độ Wide, grid row/column, pan/tilt/zoom và chất lượng H.\n\n"
        "Chú thích ảnh Wide: xanh lá = tâm PTZ chiếu qua H; cam = anchor lúc thu thập; vàng = sai khác giữa hai điểm.\n"
        "Ảnh raw gốc không bị sửa. Các điểm xanh là kết quả của ALIKED + LightGlue + MAGSAC homography.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "complete", "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
