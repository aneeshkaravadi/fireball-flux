"""Entry angle and radiant direction from the CNEOS velocity vector.

CNEOS gives the velocity at peak brightness in an Earth-centered, Earth-fixed frame: z along the
rotation axis toward the north pole, x in the equator toward the prime meridian, y completing a
right-handed set (https://cneos.jpl.nasa.gov/fireballs/intro.html). I use a spherical Earth for the
local vertical; the geodetic correction is under 0.2 degrees, well below the precision of the data.
"""
from __future__ import annotations

import numpy as np


def enu_basis(lat_deg, lon_deg):
    """Unit east, north and up vectors (each shape (..., 3)) at the given latitude and longitude."""
    la = np.radians(np.asarray(lat_deg, dtype=float))
    lo = np.radians(np.asarray(lon_deg, dtype=float))
    east = np.stack([-np.sin(lo), np.cos(lo), np.zeros_like(lo)], axis=-1)
    north = np.stack([-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)], axis=-1)
    up = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)
    return east, north, up


def entry_angle_deg(lat_deg, lon_deg, vx, vy, vz):
    """Angle of the flight path below the local horizontal, degrees. Negative means moving upward."""
    v = np.stack([np.asarray(vx, float), np.asarray(vy, float), np.asarray(vz, float)], axis=-1)
    _, _, up = enu_basis(lat_deg, lon_deg)
    speed = np.linalg.norm(v, axis=-1)
    return np.degrees(np.arcsin(-np.sum(v * up, axis=-1) / speed))


def radiant_azimuth_deg(lat_deg, lon_deg, vx, vy, vz):
    """Azimuth of the direction the object came from, degrees clockwise from north (0 to 360)."""
    v = np.stack([np.asarray(vx, float), np.asarray(vy, float), np.asarray(vz, float)], axis=-1)
    east, north, _ = enu_basis(lat_deg, lon_deg)
    heading = np.degrees(np.arctan2(np.sum(v * east, axis=-1), np.sum(v * north, axis=-1)))
    return (heading + 180.0) % 360.0


def local_solar_time_h(dates, lon_deg):
    """Local mean solar time in hours: UTC plus longitude / 15."""
    import pandas as pd

    d = pd.to_datetime(pd.Series(dates))
    utc = d.dt.hour + d.dt.minute / 60 + d.dt.second / 3600
    return ((utc.to_numpy() + np.asarray(lon_deg, dtype=float) / 15.0) % 24.0)


def isotropic_entry_angle_cdf(angle_deg):
    """Fraction of impacts shallower than `angle_deg` for an isotropic flux on a flat surface.

    The rate through a horizontal surface goes as cos(zenith) d(solid angle), so the elevation angle h
    has density sin(2h) and CDF sin^2(h). The median is 45 degrees. (Gravity bending the paths is ignored.)
    """
    return np.sin(np.radians(np.asarray(angle_deg, dtype=float))) ** 2
