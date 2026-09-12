"""
region_config.py

The registry of regions the pipeline knows how to compute for. Adding a
region means adding an entry here plus four artifact files following the
{region_id}_ prefix convention:

    {region_id}_model.pkl
    {region_id}_spi_params.pkl
    {region_id}_climatology.pkl
    {region_id}_boundary.geojson

No changes to live_status.py, refresh_job.py, or status_reader.py are
needed to add a region -- they all take region_id as a plain argument and
build paths from it.
"""

REGIONS = {
    "kobo": {
        "display_name": "Raya Kobo Woreda, North Wollo Zone, Amhara, Ethiopia",
        "short_name": "Raya Kobo",
    },
    # Structural proof-of-concept only: reuses Kobo's artifact files under a
    # second region_id to demonstrate the pipeline is genuinely parameterized,
    # not hardcoded to Kobo. This is NOT a real second region -- there is no
    # independent model, boundary, or climatology behind it, and it should
    # not be presented as one. See docs/METHODOLOGY.md.
    "demo": {
        "display_name": "Demo region (proof-of-concept duplicate of Kobo's artifacts)",
        "short_name": "Demo",
    },
}
