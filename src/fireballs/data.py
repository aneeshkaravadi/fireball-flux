"""Load the NASA JPL CNEOS fireball table (https://ssd-api.jpl.nasa.gov/doc/fireball.html).

The API returns every field as a string (or null), with latitude and longitude as positive numbers
plus a direction letter. `tidy` turns that into signed numbers in SI-friendly columns.

Field meanings, from the API documentation (version 1.2):
    date      date/time of peak brightness (GMT)
    energy    approximate total radiated energy, in units of 1e10 J
    impact-e  approximate total impact energy, kt
    alt       altitude above the geoid at peak brightness, km
    vx/vy/vz  velocity components, km/s, Earth-centered Earth-fixed frame
              (z toward the north pole, x toward the prime meridian)

The API also returns an undocumented "vel" field. The documentation says the derived velocity
magnitude was removed in version 1.1, and for a few events "vel" disagrees with the magnitude of
the documented components, so speeds here always come from vx, vy, vz.
"""
from __future__ import annotations

import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

API_URL = "https://ssd-api.jpl.nasa.gov/fireball.api"
API_VERSION = "1.2"
RAW_FIELDS = ["date", "energy", "impact-e", "lat", "lat-dir", "lon", "lon-dir", "alt", "vel", "vx", "vy", "vz"]


def parse_api(payload: dict | str) -> pd.DataFrame:
    """API JSON (dict or text) -> DataFrame of the raw string fields, one row per fireball."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    version = payload.get("signature", {}).get("version")
    if version != API_VERSION:
        warnings.warn(f"fireball API version {version!r}, this code was written for {API_VERSION}", stacklevel=2)
    if int(payload.get("count", 0)) == 0:
        return pd.DataFrame(columns=RAW_FIELDS)
    raw = pd.DataFrame(payload["data"], columns=payload["fields"])
    if len(raw) != int(payload["count"]):
        raise ValueError(f"API count {payload['count']} but {len(raw)} records")
    return raw.reindex(columns=[c for c in RAW_FIELDS if c in raw.columns])


def tidy(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw API fields -> numeric columns.

    Columns: date (UTC), radiated_J, impact_kt, lat, lon (signed degrees, N and E positive),
    alt_km, vx, vy, vz, speed (km/s, from the components), vel_api (the undocumented field).
    """
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw else pd.Series(np.nan, index=raw.index)  # noqa: E731
    out = pd.DataFrame(index=raw.index)
    out["date"] = pd.to_datetime(raw["date"], format="%Y-%m-%d %H:%M:%S")
    out["radiated_J"] = num("energy") * 1e10
    out["impact_kt"] = num("impact-e")
    lat_sign = np.where(raw.get("lat-dir") == "S", -1.0, 1.0)
    lon_sign = np.where(raw.get("lon-dir") == "W", -1.0, 1.0)
    out["lat"] = num("lat") * lat_sign
    out["lon"] = num("lon") * lon_sign
    out["alt_km"] = num("alt")
    for c in ("vx", "vy", "vz"):
        out[c] = num(c)
    out["speed"] = np.sqrt(out.vx**2 + out.vy**2 + out.vz**2)
    out["vel_api"] = num("vel")
    return out.sort_values("date").reset_index(drop=True)


def load_snapshot(path: str | Path) -> pd.DataFrame:
    """Read a CSV written by `write_snapshot` (the raw API strings) and tidy it."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    raw = raw.where(raw.notna(), None)
    return tidy(raw)


def write_snapshot(raw: pd.DataFrame, path: str | Path) -> None:
    """Save the raw API strings exactly as received (null -> empty cell)."""
    buf = io.StringIO()
    raw.to_csv(buf, index=False)
    Path(path).write_text(buf.getvalue())


def fetch(params: dict | None = None, timeout: float = 60.0) -> dict:
    """Download the full table with velocity components. Needs the network; tests never call it."""
    import requests

    params = {"vel-comp": "true", **(params or {})}
    r = requests.get(API_URL, params=params, timeout=timeout,
                     headers={"User-Agent": "fireball-flux (github.com/aneeshkaravadi/fireball-flux)"})
    r.raise_for_status()
    return r.json()
