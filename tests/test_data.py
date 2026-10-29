"""Parsing the CNEOS API response, using 12 real records saved from the API (no network)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fireballs import data

FIX = Path(__file__).parent / "fixtures" / "fireball_api_sample.json"


@pytest.fixture(scope="module")
def payload():
    return json.loads(FIX.read_text())


def test_parse_keeps_every_record_and_field(payload):
    raw = data.parse_api(payload)
    assert len(raw) == int(payload["count"]) == 12
    assert list(raw.columns) == data.RAW_FIELDS


def test_signed_coordinates_and_units(payload):
    df = data.tidy(data.parse_api(payload)).set_index("date")
    che = df.loc[pd.Timestamp("2013-02-15 03:20:26")]
    # values as printed on the CNEOS page about the event (cneos.jpl.nasa.gov/news/fireball_130301.html)
    assert che.lat == pytest.approx(54.8) and che.lon == pytest.approx(61.1)
    assert che.alt_km == pytest.approx(23.3)
    assert che.radiated_J == pytest.approx(3.75e14)
    assert che.speed == pytest.approx(18.6, abs=0.05)
    south_west = df.loc[pd.Timestamp("2026-10-04 03:14:45")]
    assert south_west.lat == pytest.approx(-41.8) and south_west.lon == pytest.approx(-173.1)


def test_missing_fields_become_nan(payload):
    df = data.tidy(data.parse_api(payload))
    no_loc = df[df.date == pd.Timestamp("2024-08-20 12:53:51")].iloc[0]
    assert np.isnan(no_loc.lat) and np.isnan(no_loc.lon) and np.isnan(no_loc.alt_km) and np.isnan(no_loc.speed)
    assert no_loc.impact_kt == pytest.approx(0.19)  # date, energy and impact-e are always defined
    assert df.impact_kt.notna().all() and df.radiated_J.notna().all()
    # where a location exists, all four location fields exist
    assert (df.lat.isna() == df.lon.isna()).all()


def test_speed_comes_from_the_components_not_the_undocumented_field(payload):
    df = data.tidy(data.parse_api(payload)).set_index("date")
    bering = df.loc[pd.Timestamp("2018-12-18 23:48:18")]
    assert bering.speed == pytest.approx(np.sqrt(6.3**2 + 3.0**2 + 31.2**2))
    assert bering.speed == pytest.approx(32.0, abs=0.05)  # JPL's 32 km/s as reported by CBC News in March 2019
    assert bering.vel_api == pytest.approx(13.6)  # the undocumented field disagrees


def test_csv_snapshot_round_trip(payload, tmp_path):
    raw = data.parse_api(payload)
    path = tmp_path / "snap.csv"
    data.write_snapshot(raw, path)
    a = data.tidy(raw)
    b = data.load_snapshot(path)
    pd.testing.assert_frame_equal(a, b)


def test_empty_result_and_version_warning():
    empty = data.parse_api({"count": 0, "signature": {"version": "1.2", "source": "NASA/JPL Fireball Data API"}})
    assert len(empty) == 0
    with pytest.warns(UserWarning, match="version"):
        data.parse_api({"count": 0, "signature": {"version": "9.9"}})
