#!/usr/bin/env python3
"""Restore a PTZ pose with short reconnect retries."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from capture_onvif import connect_onvif, read_actual_position, set_absolute_position, wait_until_position  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=80)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--pan", type=float, required=True)
    parser.add_argument("--tilt", type=float, required=True)
    parser.add_argument("--zoom", type=float, default=0.0)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=15.0)
    args = parser.parse_args()
    target = (args.pan, args.tilt, args.zoom)
    last_error = None
    for attempt in range(1, args.retries + 1):
        try:
            namespace = argparse.Namespace(host=args.host, port=args.port, user=args.user, password=args.password, onvif_timeout=args.timeout)
            _camera, _media, _profile, token, _snapshot, ptz = connect_onvif(namespace)
            request = set_absolute_position(ptz, token, *target)
            ptz.AbsoluteMove(request)
            actual = wait_until_position(ptz, token, *target, settle_seconds=2.0, timeout_seconds=args.timeout, tolerance=0.04, stable_samples=3)
            print(json.dumps({"status": "restored", "attempt": attempt, "target": target, "actual": actual}))
            return 0
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(1.0)
    print(json.dumps({"status": "failed", "target": target, "error": last_error}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
