"""
status_reader.py

The ONLY status-related module the Streamlit app imports. Deliberately
tiny: it reads the JSON file the scheduled job (refresh_job.py) writes,
wrapped in a short-TTL st.cache_data so repeated Streamlit reruns don't
re-read disk on every widget interaction. It never calls Earth Engine,
AIFS, or anything expensive -- that work happens entirely outside the
request path, in refresh_job.py.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

# How old the live cache can be before we tell the user it looks stale --
# a bit more than the job's nominal cadence, to tolerate one missed run
# without immediately alarming visitors.
STALE_AFTER_HOURS = 8


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def read_latest_status(region_id="kobo"):
    """Returns (status_dict, is_live, is_stale, error_message) for the given
    region_id. Building paths from region_id (rather than a hardcoded
    filename) is the entire mechanism that lets a second region be added
    without touching this function.

    is_live=True means the scheduled job has produced at least one result
    for this region; False means we're showing the bundled fallback
    snapshot instead (e.g. before the job has ever run).

    is_stale=True means the live cache exists but hasn't been refreshed
    recently -- worth telling the user the scheduled job may not be
    running, distinct from "no live data has ever existed."
    """
    live_path = DATA_DIR / f"{region_id}_live_status_cache.json"
    fallback_path = DATA_DIR / f"{region_id}_latest_status.json"

    try:
        status = _read_json(live_path)
        is_live = True
    except (FileNotFoundError, json.JSONDecodeError):
        try:
            status = _read_json(fallback_path)
            is_live = False
        except (FileNotFoundError, json.JSONDecodeError):
            return None, False, False, f"No status data found for region '{region_id}' — neither a live cache nor a fallback snapshot exists."

    is_stale = False
    computed_at_iso = status.get("computed_at_iso")
    if computed_at_iso:
        try:
            computed_at = pd.to_datetime(computed_at_iso)
            now = pd.Timestamp.now("UTC")
            if computed_at.tzinfo is None:
                computed_at = computed_at.tz_localize("UTC")
            age_hours = (now - computed_at).total_seconds() / 3600
            is_stale = age_hours > STALE_AFTER_HOURS
        except Exception:
            pass

    return status, is_live, is_stale, None


@st.cache_data(ttl=300)  # cheap file read -- short TTL just avoids re-reading disk on every rerun
def get_cached_status(region_id="kobo"):
    return read_latest_status(region_id)
