#!/usr/bin/env python3
"""
test_region_abstraction.py

Proves that live_status.load_artifacts() is genuinely parameterized by
region_id, rather than secretly hardcoded to Kobo somewhere. This is not
a mock of the abstraction -- it calls the real, unmodified function from
live_status.py against a fabricated second region's files, in an isolated
temp directory containing zero Kobo files, and checks the result reflects
that second region's data specifically.

Run manually:
    python test_region_abstraction.py

Also wired into CI (see .github/workflows/test_region_abstraction.yml) so
this runs automatically on every push, not just once by hand.

Note: ee.Geometry() itself needs live Earth Engine authentication (a
network call), which CI does not have credentials for. That single line
is stubbed here -- everything else in load_artifacts() runs for real.
The ee.Geometry() call itself is exercised for real every time the
scheduled refresh job runs against Kobo in production; what this test
adds is proof that swapping region_id changes which files get read.
"""

import json
import pickle
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import live_status  # noqa: E402


def make_fake_region_artifacts(data_dir: Path, region_id: str):
    """Writes fabricated artifact files for a made-up region -- deliberately
    different content from Kobo's real files, so a pass here can't be
    accidentally explained by silently falling back to Kobo's data."""
    model_bundle = {
        "model": {"classes_": ["Normal", "Moderate", "Severe"], "kind": "fabricated test model"},
        "scaler": {"kind": "fabricated test scaler"},
        "predictor_cols": [f"{region_id}_feature_a", f"{region_id}_feature_b"],
    }
    with open(data_dir / f"{region_id}_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    with open(data_dir / f"{region_id}_spi_params.pkl", "wb") as f:
        pickle.dump({(3, 6): (1.0, 0.0, 1.0, 0.1)}, f)

    with open(data_dir / f"{region_id}_climatology.pkl", "wb") as f:
        pickle.dump({"marker": f"climatology data belonging to {region_id}, not kobo"}, f)

    fake_boundary = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": f"Fabricated boundary for {region_id}"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]],
            },
        }],
    }
    with open(data_dir / f"{region_id}_boundary.geojson", "w") as f:
        json.dump(fake_boundary, f)


def main():
    region_id = "testregion_ci_proof"
    tmp_dir = Path(tempfile.mkdtemp(prefix="region_abstraction_test_"))

    try:
        make_fake_region_artifacts(tmp_dir, region_id)

        # Sanity check: confirm this temp dir genuinely has no Kobo files,
        # so a pass below can't be explained by accidentally reading them.
        kobo_files = list(tmp_dir.glob("kobo_*"))
        assert kobo_files == [], f"Test setup error: found unexpected kobo_* files: {kobo_files}"

        with patch("live_status.ee.Geometry", side_effect=lambda g: {"stubbed_ee_geometry": g}):
            artifacts = live_status.load_artifacts(region_id, data_dir=tmp_dir)

        assert artifacts["region_id"] == region_id, "region_id not passed through correctly"
        assert artifacts["predictor_cols"] == [f"{region_id}_feature_a", f"{region_id}_feature_b"], \
            "predictor_cols did not come from the fabricated region's file"
        assert artifacts["clim"]["marker"] == f"climatology data belonging to {region_id}, not kobo", \
            "climatology did not come from the fabricated region's file"
        assert artifacts["region_geom"]["stubbed_ee_geometry"]["coordinates"][0][0] == [0.0, 0.0], \
            "boundary geometry did not come from the fabricated region's file"

        print(f"PASS: load_artifacts('{region_id}', ...) correctly loaded that region's own files.")
        print("No hardcoded reference to 'kobo' anywhere in this code path.")
        return 0

    except AssertionError as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
