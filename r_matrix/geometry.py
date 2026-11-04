"""Coordinates and distances for the two geometries of the prototype.

plane   x, y are Cartesian coordinates; distance is Euclidean.
sphere  x = longitude, y = latitude, both in radians (as in the IFS ODB). Longitude may
        lie outside [-pi, pi), e.g. [0, 2 pi); it is wrapped wherever it matters.
        Points are mapped to 3-D on a sphere of radius EARTH_RADIUS (km), and distance is
        the chord between them. Gaspari-Cohn of the chordal distance is positive definite
        on the sphere, because it is positive definite in R^3. Length scales (c, h) are
        then in km.
"""
import numpy as np

EARTH_RADIUS = 6371.0      # km
GEOMETRIES = ("plane", "sphere")


def wrap_lon(lon):
    """Longitude wrapped into [-pi, pi)."""
    return np.mod(np.asarray(lon, dtype=float) + np.pi, 2.0 * np.pi) - np.pi


def points(x, y, geometry="plane"):
    """Coordinates whose Euclidean distance is the model distance: (n, 2) or (n, 3)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if geometry == "plane":
        return np.column_stack([x, y])
    if geometry == "sphere":
        cy = np.cos(y)
        return EARTH_RADIUS * np.column_stack([cy * np.cos(x), cy * np.sin(x), np.sin(y)])
    raise ValueError(f"unknown geometry {geometry!r}; expected one of {GEOMETRIES}")


def angle_to_chord(a):
    """Chord length (km) subtending central angle a (radians) on the Earth sphere."""
    return 2.0 * EARTH_RADIUS * np.sin(np.minimum(np.asarray(a, dtype=float), np.pi) / 2.0)
