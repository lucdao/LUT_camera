#!/usr/bin/env python3
"""Predict the PTZ state that places a selected Wide pixel at PTZ centre.

The script consumes either the two-stage Polynomial mapping or the
intermediate H mapping. It does not command a camera; it only prints the
predicted pan/tilt/zoom state.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

from fit_polynomial_wide_to_grid import design_matrix, normalize_points


PIPELINE_DIR = Path(__file__).resolve().parent


def latest_mapping() -> Path:
    candidates = [
        *PIPELINE_DIR.glob("run_*/03_polynomial_both_stages/polynomial_both_stages_mapping.json"),
        *PIPELINE_DIR.glob("run_*/02_aliked_lightglue_wide_ptz_mapping/wide_pixel_to_ptz_mapping.json"),
    ]
    if not candidates:
        raise RuntimeError("Không tìm thấy wide_pixel_to_ptz_mapping.json.")
    return max(candidates, key=lambda item: item.stat().st_mtime)


class PoseSurface:
    def __init__(self, grid: np.ndarray, method: str, coefficients: np.ndarray) -> None:
        self.grid = grid
        self.method = method
        self.coefficients = coefficients
        self.rows, self.columns, _ = grid.shape
        self.u_values = np.arange(self.columns, dtype=np.float64)
        self.v_values = np.arange(self.rows, dtype=np.float64)

    def predict(self, u: float, v: float) -> np.ndarray:
        if self.method == "affine":
            return np.array([1.0, u, v], dtype=np.float64) @ self.coefficients
        result = np.empty(3, dtype=np.float64)
        for axis in range(3):
            row_values = np.array(
                [PchipInterpolator(self.u_values, self.grid[row, :, axis], extrapolate=True)(u) for row in range(self.rows)],
                dtype=np.float64,
            )
            result[axis] = float(PchipInterpolator(self.v_values, row_values, extrapolate=True)(v))
        return result


def transform_point(point: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(point.astype(np.float32).reshape(1, 1, 2), homography).reshape(2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, default=None, help="Defaults to the latest completed mapping.")
    parser.add_argument("--x", type=float, required=True, help="Wide pixel x.")
    parser.add_argument("--y", type=float, required=True, help="Wide pixel y.")
    parser.add_argument("--no-clamp", action="store_true", help="Allow extrapolation beyond the calibrated 5x6 lattice.")
    args = parser.parse_args()

    path = (args.mapping or latest_mapping()).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    model = data.get("model") or data.get("global_mapping")
    if model is None:
        raise RuntimeError("Mapping không có model hoặc global_mapping.")
    grid = np.asarray(model["state_grid_actual_after_capture"], dtype=np.float64)
    coefficients = np.asarray(model["linearity_probe"]["affine_coefficients_intercept_u_v"], dtype=np.float64)
    surface = PoseSurface(grid, model["state_surface"], coefficients)
    point = np.asarray([[args.x, args.y]], dtype=np.float64)
    if "H_wide_pixel_to_ptz_lattice" in model:
        raw_uv = transform_point(point[0], np.asarray(model["H_wide_pixel_to_ptz_lattice"], dtype=np.float64))
    else:
        powers = [tuple(pair) for pair in model["terms_powers_xy"]]
        normalized = normalize_points(point, int(model["input_normalization"]["width"]), int(model["input_normalization"]["height"]))
        raw_uv = (design_matrix(normalized, powers) @ np.asarray(model["coefficients_uv"], dtype=np.float64))[0]
    used_u, used_v = float(raw_uv[0]), float(raw_uv[1])
    clamped = False
    if not args.no_clamp:
        clamped_u = min(max(used_u, 0.0), surface.columns - 1.0)
        clamped_v = min(max(used_v, 0.0), surface.rows - 1.0)
        clamped = not (math.isclose(used_u, clamped_u) and math.isclose(used_v, clamped_v))
        used_u, used_v = clamped_u, clamped_v
    pan, tilt, zoom = surface.predict(used_u, used_v)
    print(
        json.dumps(
            {
                "mapping": str(path),
                "wide_pixel": [args.x, args.y],
                "lattice_uv_raw": raw_uv.tolist(),
                "lattice_uv_used": [used_u, used_v],
                "clamped_to_calibrated_lattice": clamped,
                "pan": float(pan),
                "tilt": float(tilt),
                "zoom": float(zoom),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
