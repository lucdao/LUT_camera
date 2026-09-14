#!/usr/bin/env python3
"""Move PTZ to 30 Polynomial predictions, capture, and mark PTZ centres."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import RTSPFrameReader, read_actual_position, set_absolute_position  # noqa: E402
from collect_ptz_pan_tilt_grid import (  # noqa: E402
    discover_onvif_ranges,
    normalize_position,
    read_and_normalize_after_capture,
    wait_for_command_pose,
)
from execute_random_wide_point_validation import (  # noqa: E402
    build_ptz_rtsp_url,
    capture_frame,
    close_rtsp_reader,
    connect_ptz,
    redact_url,
)
from fit_polynomial_wide_to_grid import (  # noqa: E402
    PoseSurface,
    design_matrix,
    normalize_points,
    pose_from_grid,
    term_powers,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_predictions(mapping_path: Path, source_mapping_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    source_mapping = json.loads(source_mapping_path.read_text(encoding="utf-8"))
    pairs = sorted(source_mapping.get("centre_pairs", []), key=lambda item: int(item["ptz_index"]))
    if len(pairs) != 30:
        raise RuntimeError(f"Cần đúng 30 anchor Wide, hiện có {len(pairs)}.")
    if "model" in mapping:
        model = mapping["model"]
    elif "global_mapping" in mapping:
        # The both-stages experiment stores its runtime Wide->grid model in
        # ``global_mapping`` and the pose-surface linearity coefficients in
        # the source mapping's model.
        model = dict(mapping["global_mapping"])
        source_model = source_mapping["model"]
        model["linearity_probe"] = source_model["linearity_probe"]
    else:
        raise RuntimeError("Mapping không có model hoặc global_mapping để điều khiển PTZ.")
    powers = [tuple(pair) for pair in model["terms_powers_xy"]]
    coefficients = np.asarray(model["coefficients_uv"], dtype=np.float64)
    width = int(model["input_normalization"]["width"])
    height = int(model["input_normalization"]["height"])
    grid = np.asarray(model["state_grid_actual_after_capture"], dtype=np.float64)
    affine = np.asarray(model["linearity_probe"]["affine_coefficients_intercept_u_v"], dtype=np.float64)
    surface = PoseSurface(grid, model["state_surface"], affine)
    anchors = np.asarray([pair["expected_wide_anchor"] for pair in pairs], dtype=np.float64)
    normalized = normalize_points(anchors, width, height)
    features = design_matrix(normalized, powers)
    lattice = features @ coefficients
    predictions: list[dict[str, Any]] = []
    state_min = np.nanmin(grid, axis=(0, 1))
    state_max = np.nanmax(grid, axis=(0, 1))
    for index, (pair, point, uv) in enumerate(zip(pairs, anchors, lattice, strict=True)):
        pose = pose_from_grid(uv, surface, clamp=True)
        predicted = np.asarray([pose["pan"], pose["tilt"], pose["zoom"]], dtype=np.float64)
        predicted = np.clip(predicted, state_min, state_max)
        predictions.append(
            {
                "sample_index": index,
                "ptz_index": int(pair["ptz_index"]),
                "row": int(pair["row"]),
                "column": int(pair["column"]),
                "wide_pixel": [int(round(point[0])), int(round(point[1]))],
                "polynomial_lattice_uv_raw": [float(uv[0]), float(uv[1])],
                "polynomial_lattice_uv_used": pose["lattice_uv_used"],
                "clamped_to_grid": bool(pose["clamped"]),
                "predicted": {"pan": float(predicted[0]), "tilt": float(predicted[1]), "zoom": float(predicted[2])},
            }
        )
    return predictions, model


def mark_wide(image_path: Path, output_path: Path, points: list[dict[str, Any]], source_size: tuple[int, int]) -> list[list[int]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được Wide image: {image_path}")
    height, width = image.shape[:2]
    source_width, source_height = source_size
    current_points: list[list[int]] = []
    for item in points:
        source_x, source_y = item["wide_pixel"]
        x = min(width - 1, max(0, int(round(float(source_x) * width / source_width))))
        y = min(height - 1, max(0, int(round(float(source_y) * height / source_height))))
        current_points.append([x, y])
        cv2.circle(image, (x, y), 12, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(image, (x, y), (0, 255, 255), cv2.MARKER_TILTED_CROSS, 28, 3, cv2.LINE_AA)
        cv2.putText(image, f"{item['sample_index']:02d}", (x + 14, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, "30 Wide targets for Polynomial PTZ validation", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
        raise RuntimeError(f"Không ghi được Wide marked image: {output_path}")
    return current_points


def write_ptz_preview(raw_path: Path, preview_path: Path, item: dict[str, Any]) -> None:
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Không đọc được PTZ raw image: {raw_path}")
    height, width = image.shape[:2]
    centre = (width // 2, height // 2)
    cv2.drawMarker(image, centre, (0, 0, 255), cv2.MARKER_CROSS, max(40, min(width, height) // 8), 5, cv2.LINE_AA)
    cv2.circle(image, centre, max(18, min(width, height) // 32), (0, 0, 255), 3, cv2.LINE_AA)
    predicted = item["commanded_target"]
    cv2.putText(image, f"Wide point {item['sample_index']:02d}", (35, height - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"pan={predicted['pan']:.4f} tilt={predicted['tilt']:.4f} zoom={predicted['zoom']:.4f}", (35, height - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(preview_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
        raise RuntimeError(f"Không ghi được PTZ preview: {preview_path}")


def make_contact_sheet(paths: list[Path], output_path: Path, columns: int = 5) -> None:
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
    images = [image for image in images if image is not None]
    if not images:
        return
    cell_width = 480
    cell_height = int(round(images[0].shape[0] * cell_width / images[0].shape[1]))
    rows = (len(images) + columns - 1) // columns
    sheet = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
    for index, image in enumerate(images):
        thumb = cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        sheet[row * cell_height : (row + 1) * cell_height, column * cell_width : (column + 1) * cell_width] = thumb
    cv2.imwrite(str(output_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--source-mapping", type=Path, required=True)
    parser.add_argument("--wide-url", default=os.getenv("WIDE_RTSP_URL"))
    parser.add_argument("--ptz-host", default="192.168.1.8")
    parser.add_argument("--ptz-port", type=int, default=80)
    parser.add_argument("--ptz-user", default="admin")
    parser.add_argument("--ptz-password", default=os.getenv("PTZ_PASSWORD"))
    parser.add_argument("--ptz-rtsp-url", default=os.getenv("PTZ_RTSP_URL"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--rtsp-timeout", type=float, default=15.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=45.0)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if not args.wide_url:
        raise SystemExit("Thiếu WIDE_RTSP_URL hoặc --wide-url.")
    if not args.ptz_password and not args.ptz_rtsp_url:
        raise SystemExit("Thiếu PTZ_PASSWORD hoặc --ptz-rtsp-url.")

    predictions, mapping_model = load_predictions(args.mapping.resolve(), args.source_mapping.resolve())
    output_dir = (args.output_dir or args.mapping.resolve().parent / f"04_execute_polynomial_30_points_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"Output đã có dữ liệu, không ghi đè: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    wide_dir = output_dir / "wide"
    raw_dir = output_dir / "ptz_raw"
    preview_dir = output_dir / "ptz_preview_centres"
    for directory in (wide_dir, raw_dir, preview_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "execution_metadata.json"
    metadata: dict[str, Any] = {
        "stage": "execute_polynomial_wide_point_validation",
        "pipeline": "Polynomial Wide pixel -> PTZ grid -> existing pose surface",
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "mapping": str(args.mapping.resolve()),
        "source_mapping": str(args.source_mapping.resolve()),
        "wide_stream_url": redact_url(args.wide_url),
        "ptz": {"host": args.ptz_host, "port": args.ptz_port, "user": args.ptz_user},
        "points": predictions,
        "wide_capture": None,
        "captures": [],
        "failures": [],
    }
    write_json(metadata_path, metadata)

    wide_reader: RTSPFrameReader | None = None
    ptz_reader: RTSPFrameReader | None = None
    preview_paths: list[Path] = []
    source_size = (
        int(mapping_model["input_normalization"]["width"]),
        int(mapping_model["input_normalization"]["height"]),
    )
    try:
        wide_raw = wide_dir / "wide_current.jpg"
        wide_marked = wide_dir / "wide_30_polynomial_points_marked.jpg"
        wide_reader = RTSPFrameReader(args.wide_url, args.rtsp_timeout)
        wide_width, wide_height = capture_frame(wide_reader, wide_raw, args.rtsp_timeout)
        close_rtsp_reader(wide_reader)
        wide_reader = None
        current_points = mark_wide(wide_raw, wide_marked, predictions, source_size)
        metadata["wide_capture"] = {
            "captured_at_utc": utc_now(),
            "raw_image": str(wide_raw.relative_to(output_dir)),
            "marked_image": str(wide_marked.relative_to(output_dir)),
            "width": wide_width,
            "height": wide_height,
            "source_mapping_size": list(source_size),
        }
        for index, point in enumerate(current_points):
            metadata["points"][index]["wide_pixel_current_capture"] = point
        write_json(metadata_path, metadata)

        camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
        ranges = discover_onvif_ranges(ptz, profile)
        metadata["onvif_position_ranges"] = ranges
        rtsp_url = build_ptz_rtsp_url(args)
        ptz_reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
        write_json(metadata_path, metadata)

        for point in predictions:
            index = int(point["sample_index"])
            commanded = point["predicted"]
            raw_path = raw_dir / f"ptz_{index:02d}.jpg"
            preview_path = preview_dir / f"ptz_{index:02d}_centre.jpg"
            success = False
            for attempt in range(1, args.retries + 2):
                try:
                    request = set_absolute_position(ptz, token, commanded["pan"], commanded["tilt"], commanded["zoom"])
                    ptz.AbsoluteMove(request)
                    before = wait_for_command_pose(
                        ptz,
                        token,
                        commanded,
                        ranges,
                        settle_seconds=args.settle,
                        timeout_seconds=args.move_timeout,
                        tolerance=args.tolerance,
                        stable_samples=args.stable_samples,
                    )
                    width, height = capture_frame(ptz_reader, raw_path, args.rtsp_timeout)
                    after = read_and_normalize_after_capture(ptz, token, commanded, ranges, args.onvif_timeout, args.tolerance)
                    record = {
                        **point,
                        "status": "ok",
                        "attempt": attempt,
                        "captured_at_utc": utc_now(),
                        "commanded_target": commanded,
                        "actual_before_capture": before["normalized"],
                        "actual_before_capture_raw": before["raw"],
                        "actual_before_capture_normalization": before["normalization"],
                        "actual_before_capture_wait_fallback": before["wait_fallback"],
                        "actual_after_capture": after["normalized"],
                        "actual_after_capture_raw": after["raw"],
                        "actual_after_capture_normalization": after["normalization"],
                        "raw_image": str(raw_path.relative_to(output_dir)),
                        "preview_image": str(preview_path.relative_to(output_dir)),
                        "width": width,
                        "height": height,
                    }
                    write_ptz_preview(raw_path, preview_path, record)
                    metadata["captures"].append(record)
                    preview_paths.append(preview_path)
                    success = True
                    print(f"{index + 1:02d}/30 captured", flush=True)
                    break
                except Exception as exc:  # noqa: BLE001 - persist every failed pose
                    if attempt <= args.retries:
                        time.sleep(min(5.0, float(attempt)))
                        try:
                            close_rtsp_reader(ptz_reader)
                            camera, media, profile, token, snapshot_uri, ptz = connect_ptz(args)
                            ranges = discover_onvif_ranges(ptz, profile)
                            ptz_reader = RTSPFrameReader(rtsp_url, args.rtsp_timeout)
                        except Exception:
                            pass
                    else:
                        metadata["failures"].append(
                            {
                                **point,
                                "status": "error",
                                "attempts": attempt,
                                "failed_at_utc": utc_now(),
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        print(f"{index + 1:02d}/30 failed: {type(exc).__name__}", flush=True)
                        if not args.continue_on_error:
                            raise
            write_json(metadata_path, metadata)
            if not success and not args.continue_on_error:
                break
    finally:
        close_rtsp_reader(wide_reader)
        close_rtsp_reader(ptz_reader)
        make_contact_sheet(preview_paths, output_dir / "ptz_30_centres_contact_sheet.jpg")
        metadata["finished_at_utc"] = utc_now()
        metadata["status"] = "complete" if len(metadata["captures"]) == 30 else "incomplete"
        metadata["summary"] = {"requested": 30, "captured": len(metadata["captures"]), "failed": len(metadata["failures"])}
        write_json(metadata_path, metadata)

    print(json.dumps({"status": metadata["status"], "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)
    return 0 if metadata["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
