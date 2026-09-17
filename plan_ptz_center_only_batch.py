#!/usr/bin/env python3
"""Plan ten centre-only PTZ tests, including five projected outside Wide."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from build_ptz_pixel_motion_relation import load_entries, project  # noqa: E402
from test_ptz_motion_relation_live import load_json, predict_delta  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relation-dir", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    relation = load_json(args.relation_dir / "ptz_motion_relation_models.json")
    entries, _source, ptz_size, wide_size = load_entries(args.mapping_dir)
    model = relation["models"]["quadratic_no_intercept"]
    ptz_width, ptz_height = ptz_size
    wide_width, wide_height = wide_size
    pan_values = [entry["pan"] for entry in entries]
    tilt_values = [entry["tilt"] for entry in entries]
    pan_range = [min(pan_values), max(pan_values)]
    tilt_range = [min(tilt_values), max(tilt_values)]

    x_values = [100.0, 300.0, 550.0, 800.0, 1280.0, 1760.0, 2010.0, 2260.0, 2460.0]
    y_values = [100.0, 360.0, 720.0, 1080.0, 1340.0]
    candidates: list[dict[str, Any]] = []
    for entry in entries:
        for y in y_values:
            for x in x_values:
                point = np.array([x, y], dtype=np.float64)
                error = np.array([ptz_width / 2.0 - x, ptz_height / 2.0 - y])
                delta = predict_delta(model, float(error[0]), float(error[1]), ptz_width, ptz_height)
                target = np.array([entry["pan"] + delta[0], entry["tilt"] + delta[1]])
                wide_point = project(entry["H"], point.reshape(1, 2))[0]
                outside_wide = bool(
                    wide_point[0] < 0.0 or wide_point[0] >= wide_width
                    or wide_point[1] < 0.0 or wide_point[1] >= wide_height
                )
                inside_device = bool(np.all(target >= -0.99) and np.all(target <= 0.99))
                if not inside_device:
                    continue
                outside_sampled = bool(
                    target[0] < pan_range[0] or target[0] > pan_range[1]
                    or target[1] < tilt_range[0] or target[1] > tilt_range[1]
                )
                candidates.append({
                    "source_ptz_index": entry["ptz_index"],
                    "source_row": entry["row"],
                    "source_column": entry["column"],
                    "source_position": [entry["pan"], entry["tilt"], entry["zoom"]],
                    "pixel": [x, y],
                    "error_to_centre_px": error.tolist(),
                    "delta_pan": float(delta[0]),
                    "delta_tilt": float(delta[1]),
                    "predicted_destination": [float(target[0]), float(target[1]), entry["zoom"]],
                    "projected_wide_pixel": wide_point.tolist(),
                    "outside_wide": outside_wide,
                    "outside_sampled_ptz_range": outside_sampled,
                })

    def select(pool: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        chosen: list[dict[str, Any]] = []
        used_sources: set[int] = set()
        for item in sorted(pool, key=lambda value: value["source_ptz_index"]):
            if item["source_ptz_index"] in used_sources:
                continue
            chosen.append(item)
            used_sources.add(item["source_ptz_index"])
            if len(chosen) == count:
                return chosen
        for item in pool:
            if item in chosen:
                continue
            chosen.append(item)
            if len(chosen) == count:
                return chosen
        return chosen

    outside = [item for item in candidates if item["outside_wide"]]
    inside = [item for item in candidates if not item["outside_wide"]]
    outside_selected = select(outside, 5)
    inside_selected = select(inside, 5)
    if len(outside_selected) < 5 or len(inside_selected) < 5:
        raise RuntimeError(f"Không đủ mẫu: outside={len(outside_selected)}, inside={len(inside_selected)}")
    cases = []
    for number, item in enumerate(outside_selected + inside_selected, start=1):
        case = dict(item)
        case["case_id"] = f"case_{number:02d}"
        case["category"] = "outside_wide" if number <= 5 else "inside_wide_control"
        cases.append(case)
    payload = {
        "ptz_image_size": {"width": ptz_width, "height": ptz_height},
        "wide_image_size": {"width": wide_width, "height": wide_height},
        "sampled_pan_range": pan_range,
        "sampled_tilt_range": tilt_range,
        "outside_definition": "H_source(pixel) lies outside the Wide image bounds",
        "case_count": len(cases),
        "outside_wide_count": sum(case["category"] == "outside_wide" for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "complete", "case_count": len(cases), "outside_wide_count": payload["outside_wide_count"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
