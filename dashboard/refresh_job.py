#!/usr/bin/env python3
"""
refresh_job.py

Standalone script that runs the full Earth Engine + AIFS ensemble
computation and writes the result to disk as {region_id}_live_status_cache.json.

This is intended to run on a schedule (e.g. a GitHub Actions cron job),
completely decoupled from the Streamlit app's request path -- the app
(via status_reader.py) only ever reads the file this script writes. No
visitor to the dashboard ever triggers the expensive Earth Engine/AIFS
pull themselves.

Usage:
    python refresh_job.py --region kobo

Requires the GEE service account credentials as a JSON string in the
GEE_SERVICE_ACCOUNT_JSON environment variable -- this is deliberately NOT
st.secrets, since this script runs outside Streamlit entirely (e.g. as a
GitHub Actions step, or any other cron runner).
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import live_status  # noqa: E402
from region_config import REGIONS  # noqa: E402

DATA_DIR = Path(__file__).parent / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--region", default="kobo",
        help=f"Region ID to refresh. Known regions: {list(REGIONS)}",
    )
    args = parser.parse_args()

    if args.region not in REGIONS:
        print(f"ERROR: unknown region '{args.region}'. Known regions: {list(REGIONS)}", file=sys.stderr)
        sys.exit(1)

    sa_json = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    if not sa_json:
        print("ERROR: GEE_SERVICE_ACCOUNT_JSON environment variable not set.", file=sys.stderr)
        sys.exit(1)

    try:
        sa_info = json.loads(sa_json)
    except json.JSONDecodeError as e:
        print(f"ERROR: GEE_SERVICE_ACCOUNT_JSON is not valid JSON: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        status = live_status.compute_status(args.region, sa_info, DATA_DIR)
    except Exception as e:
        print(f"ERROR: refresh job failed for region '{args.region}': {e}", file=sys.stderr)
        sys.exit(1)

    output_path = DATA_DIR / f"{args.region}_live_status_cache.json"
    with open(output_path, "w") as f:
        json.dump(status, f, indent=2)

    print(f"OK: wrote {output_path}")
    print(f"Status: {status['status']}, class: {status['current_class']}, computed_at: {status['computed_at']}")


if __name__ == "__main__":
    main()

