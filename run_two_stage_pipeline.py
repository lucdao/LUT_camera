#!/usr/bin/env python3
"""One-command runner for the complete two-stage Polynomial pipeline.

Default execution is offline and starts from an already captured run. Add
``--capture`` to collect a new unknown-camera dynamic grid first. Add
``--execute`` only when the fitted model should command the real PTZ to the
30 generated Wide points.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def redacted(value: str | None) -> str | None:
    if not value:
        return value
    if "@" in value and "://" in value:
        prefix, rest = value.split("://", 1)
        if "@" in rest and ":" in rest.split("@", 1)[0]:
            user = rest.split("@", 1)[0].split(":", 1)[0]
            return f"{prefix}://{user}:***@{rest.split('@', 1)[1]}"
    return value


def run_step(name: str, command: list[str], env: dict[str, str], log_path: Path) -> None:
    print(f"[{name}] {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=PACKAGE_DIR, env=env, text=True, capture_output=True, check=False)
    log_path.write_text(
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )
    if result.stdout:
        print(result.stdout.rstrip(), flush=True)
    if result.returncode != 0:
        if result.stderr:
            print(result.stderr.rstrip(), file=sys.stderr, flush=True)
        raise RuntimeError(f"Stage {name} thất bại, exit code {result.returncode}.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=None, help="Run đã capture; khi --capture bỏ trống sẽ tự tạo run mới.")
    parser.add_argument("--capture", action="store_true", help="Capture Wide + dynamic PTZ grid trước khi fit.")
    parser.add_argument("--execute", action="store_true", help="Sau khi fit, điều khiển PTZ tới 30 điểm Wide.")
    parser.add_argument("--wide-url", default=os.getenv("WIDE_RTSP_URL"))
    parser.add_argument("--wide-count", type=int, default=8)
    parser.add_argument("--wide-interval", type=float, default=0.5)
    parser.add_argument("--pan-count", type=int, default=30)
    parser.add_argument("--tilt-count", type=int, default=6)
    parser.add_argument("--pans", default=None, help="Pan ONVIF values, comma-separated; normally omit for camera-derived range.")
    parser.add_argument("--tilts", default=None, help="Tilt ONVIF values, comma-separated; normally omit for camera-derived range.")
    parser.add_argument("--zoom", type=float, default=None)
    parser.add_argument("--ptz-host", default="192.168.1.8")
    parser.add_argument("--ptz-port", type=int, default=80)
    parser.add_argument("--ptz-user", default="admin")
    parser.add_argument("--ptz-password", default=os.getenv("PTZ_PASSWORD"))
    parser.add_argument("--ptz-rtsp-url", default=os.getenv("PTZ_RTSP_URL"))
    parser.add_argument("--onvif-timeout", type=float, default=10.0)
    parser.add_argument("--rtsp-timeout", type=float, default=15.0)
    parser.add_argument("--settle", type=float, default=2.0)
    parser.add_argument("--move-timeout", type=float, default=45.0)
    parser.add_argument("--tolerance", type=float, default=0.04)
    parser.add_argument("--stable-samples", type=int, default=3)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--ransac-threshold", type=float, default=7.0)
    parser.add_argument("--min-inliers", type=int, default=12)
    parser.add_argument("--min-inlier-ratio", type=float, default=0.18)
    parser.add_argument("--max-median-reprojection-error", type=float, default=7.0)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--max-area-ratio", type=float, default=1.2)
    parser.add_argument("--center-boundary-margin", type=float, default=200.0)
    parser.add_argument("--lattice-ransac-threshold", type=float, default=0.45)
    parser.add_argument("--pchip-trigger", type=float, default=0.0015)
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--overwrite-mapping", action="store_true")
    parser.add_argument("--local-degree", type=int, default=2, choices=(2,))
    parser.add_argument("--local-ransac-threshold-px", type=float, default=7.0)
    parser.add_argument("--global-ransac-threshold-grid", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.capture and not args.wide_url:
        raise SystemExit("--capture cần WIDE_RTSP_URL hoặc --wide-url.")
    if args.capture and not args.ptz_password and not args.ptz_rtsp_url:
        raise SystemExit("--capture cần PTZ_PASSWORD hoặc --ptz-rtsp-url.")
    if args.capture and (args.pan_count != 30 or args.tilt_count != 6):
        raise SystemExit("Bản fit hai tầng hiện yêu cầu grid chuẩn 30 pan x 6 tilt = 180 ảnh.")
    if args.execute and (not args.wide_url or (not args.ptz_password and not args.ptz_rtsp_url)):
        raise SystemExit("--execute cần WIDE_RTSP_URL và PTZ_PASSWORD/PTZ_RTSP_URL.")
    if not args.capture and not args.run_root:
        raise SystemExit("Không có --capture: cần truyền --run-root của một run đã complete.")

    run_root = (args.run_root or PACKAGE_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}").resolve()
    if args.capture and run_root.exists() and any(run_root.iterdir()):
        raise SystemExit(f"Run output đã có dữ liệu, không ghi đè: {run_root}")
    if not args.capture and not run_root.is_dir():
        raise SystemExit(f"Không tìm thấy run-root: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)
    log_dir = run_root / "pipeline_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    execution: dict[str, Any] = {
        "pipeline": "unknown_camera_two_stage_polynomial",
        "status": "running",
        "started_at_utc": utc_now(),
        "finished_at_utc": None,
        "run_root": str(run_root),
        "options": {
            key: redacted(str(value)) if key in {"wide_url", "ptz_rtsp_url"} else str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"ptz_password"}
        },
        "stages": [],
    }
    manifest_path = run_root / "two_stage_pipeline_metadata.json"
    write_json(manifest_path, execution)
    env = os.environ.copy()
    if args.wide_url:
        env["WIDE_RTSP_URL"] = args.wide_url
    if args.ptz_password:
        env["PTZ_PASSWORD"] = args.ptz_password
    if args.ptz_rtsp_url:
        env["PTZ_RTSP_URL"] = args.ptz_rtsp_url

    try:
        if args.capture:
            capture_command = [
                sys.executable, str(PACKAGE_DIR / "collect_full_dynamic_pipeline.py"),
                "--run-root", str(run_root),
                "--wide-count", str(args.wide_count), "--wide-interval", str(args.wide_interval),
                "--pan-count", str(args.pan_count), "--tilt-count", str(args.tilt_count),
                "--ptz-host", args.ptz_host, "--ptz-port", str(args.ptz_port), "--ptz-user", args.ptz_user,
                "--onvif-timeout", str(args.onvif_timeout), "--rtsp-timeout", str(args.rtsp_timeout),
                "--settle", str(args.settle), "--move-timeout", str(args.move_timeout),
                "--tolerance", str(args.tolerance), "--stable-samples", str(args.stable_samples),
                "--retries", str(args.retries),
            ]
            for option, value in (("--pans", args.pans), ("--tilts", args.tilts), ("--zoom", args.zoom)):
                if value is not None:
                    capture_command.extend([option, str(value)])
            if args.continue_on_error:
                capture_command.append("--continue-on-error")
            run_step("capture", capture_command, env, log_dir / "01_capture.log")
            execution["stages"].append({"name": "capture", "status": "complete"})
            write_json(manifest_path, execution)

        mapping_dir = run_root / "02_aliked_lightglue_wide_ptz_mapping"
        build_command = [
            sys.executable, str(PACKAGE_DIR / "build_wide_ptz_mapping.py"),
            "--run-root", str(run_root), "--output-dir", str(mapping_dir),
            "--device", args.device, "--max-side", str(args.max_side), "--max-keypoints", str(args.max_keypoints),
            "--ransac-threshold", str(args.ransac_threshold), "--min-inliers", str(args.min_inliers),
            "--min-inlier-ratio", str(args.min_inlier_ratio),
            "--max-median-reprojection-error", str(args.max_median_reprojection_error),
            "--min-area-ratio", str(args.min_area_ratio), "--max-area-ratio", str(args.max_area_ratio),
            "--center-boundary-margin", str(args.center_boundary_margin),
            "--lattice-ransac-threshold", str(args.lattice_ransac_threshold),
            "--pchip-trigger", str(args.pchip_trigger), "--sample-count", str(args.sample_count), "--seed", str(args.seed),
        ]
        if args.overwrite_mapping:
            build_command.append("--overwrite")
        run_step("feature_match_and_homography", build_command, env, log_dir / "02_feature_match_and_homography.log")
        execution["stages"].append({"name": "ALIKED + LightGlue + MAGSAC", "status": "complete", "output": str(mapping_dir)})
        write_json(manifest_path, execution)

        polynomial_dir = run_root / "03_polynomial_both_stages"
        polynomial_command = [
            sys.executable, str(PACKAGE_DIR / "fit_polynomial_both_stages.py"),
            "--run-root", str(run_root), "--output-dir", str(polynomial_dir),
            "--local-degree", str(args.local_degree),
            "--local-ransac-threshold-px", str(args.local_ransac_threshold_px),
            "--global-ransac-threshold-grid", str(args.global_ransac_threshold_grid),
            "--seed", str(args.seed),
        ]
        run_step("polynomial_stage_1_and_2", polynomial_command, env, log_dir / "03_polynomial_both_stages.log")
        polynomial_mapping = polynomial_dir / "polynomial_both_stages_mapping.json"
        execution["stages"].append({"name": "Polynomial local PTZ->Wide + Polynomial global Wide->grid", "status": "complete", "output": str(polynomial_mapping)})
        write_json(manifest_path, execution)

        if args.execute:
            execute_dir = run_root / "04_execute_polynomial_30_points"
            execute_command = [
                sys.executable, str(PACKAGE_DIR / "execute_polynomial_wide_point_validation.py"),
                "--mapping", str(polynomial_mapping),
                "--source-mapping", str(mapping_dir / "wide_pixel_to_ptz_mapping.json"),
                "--ptz-host", args.ptz_host, "--ptz-port", str(args.ptz_port), "--ptz-user", args.ptz_user,
                "--output-dir", str(execute_dir), "--onvif-timeout", str(args.onvif_timeout),
                "--rtsp-timeout", str(args.rtsp_timeout), "--settle", str(args.settle),
                "--move-timeout", str(args.move_timeout), "--tolerance", str(args.tolerance),
                "--stable-samples", str(args.stable_samples), "--retries", str(args.retries),
                "--continue-on-error",
            ]
            run_step("execute_30_points", execute_command, env, log_dir / "04_execute_30_points.log")
            execution["stages"].append({"name": "PTZ execution and 30 centre previews", "status": "complete", "output": str(execute_dir)})
        execution["status"] = "complete"
        return_code = 0
    except Exception as exc:  # noqa: BLE001 - persist actionable failure state
        execution["status"] = "failed"
        execution["error_type"] = type(exc).__name__
        execution["error"] = str(exc)
        return_code = 1
    finally:
        execution["finished_at_utc"] = utc_now()
        write_json(manifest_path, execution)
    print(json.dumps({"status": execution["status"], "run_root": str(run_root)}, ensure_ascii=False), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
