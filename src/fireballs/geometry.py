"""Entry angle and radiant direction from the CNEOS velocity vector.

CNEOS gives the velocity at peak brightness in an Earth-centered, Earth-fixed frame: z along the
rotation axis toward the north pole, x in the equator toward the prime meridian, y completing a
right-handed set (https://cneos.jpl.nasa.gov/fireballs/intro.html). I use a spherical Earth for the
local vertical; the geodetic correction is under 0.2 degrees, well below the precision of the data.

An angle measured against the local horizontal belongs to one point on the path. A straight path
over a curved Earth meets the horizontal at a different angle at every altitude: (R + z) cos(angle)
is the distance of the line from the Earth's center at its closest point, so it stays the same all
along the line. Over the 250 km of path between 97 km and 23 km altitude, Chelyabinsk's angle changes
by more than 2 degrees. So two angles can only be compared at the same point on the path.
"""
from __future__ import annotations

import numpy as np

R_EARTH_KM = 6371.0     # mean radius of the spherical Earth used here



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


def angle_at_altitude(angle_deg, alt_from_km, alt_to_km, radius_km: float = R_EARTH_KM):
    """Entry angle of the same straight path at another altitude ((R + z) cos(angle) is constant).

    nan where the line never gets down to `alt_to_km`.
    """
    c = (radius_km + np.asarray(alt_from_km, float)) * np.cos(np.radians(angle_deg)) / (radius_km + np.asarray(alt_to_km, float))
    with np.errstate(invalid="ignore"):
        return np.where(c <= 1.0, np.degrees(np.arccos(np.minimum(c, 1.0))), np.nan)


def trace_to_altitude(lat_deg, lon_deg, alt_km, vx, vy, vz, target_alt_km, radius_km: float = R_EARTH_KM):
    """Follow the straight line through a reported point up (backward in time) or down to `target_alt_km`.

    Returns the latitude, longitude, entry angle and radiant azimuth (degrees) at that point, on a
    spherical Earth. The velocity vector itself doesn't change along a straight line; only the local
    vertical does.
    """
    v = np.stack([np.asarray(vx, float), np.asarray(vy, float), np.asarray(vz, float)], axis=-1)
    _, _, up = enu_basis(lat_deg, lon_deg)
    r0 = (radius_km + np.asarray(alt_km, float))[..., None] * up
    vhat = v / np.linalg.norm(v, axis=-1, keepdims=True)
    rising = np.asarray(target_alt_km, float) > np.asarray(alt_km, float)
    d = np.where(rising[..., None], -vhat, vhat)
    b = np.sum(r0 * d, axis=-1)
    c = np.sum(r0 * r0, axis=-1) - (radius_km + np.asarray(target_alt_km, float)) ** 2
    with np.errstate(invalid="ignore"):
        root = np.sqrt(b * b - c)
    t = np.where(rising, -b + root, -b - root)
    r1 = r0 + t[..., None] * d
    lat1 = np.degrees(np.arcsin(r1[..., 2] / np.linalg.norm(r1, axis=-1)))
    lon1 = np.degrees(np.arctan2(r1[..., 1], r1[..., 0]))
    return lat1, lon1, entry_angle_deg(lat1, lon1, vx, vy, vz), radiant_azimuth_deg(lat1, lon1, vx, vy, vz)


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
