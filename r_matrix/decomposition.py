"""Spatial domain and its regular Px x Py decomposition (deck slide 3).

The rectangle [xmin, xmax] x [ymin, ymax] is split into Px columns and Py rows of
equal cells, one per MPI task. Cells are numbered row-major::

        j = 2 |  8  9 10 11
        j = 1 |  4  5  6  7          owner = i + Px * j
        j = 0 |  0  1  2  3
              +-------------
                 i = 0 1 2 3

Ownership contract, which makes the owner of every point unique and deterministic:

* i = floor((x - xmin) / dx),  j = floor((y - ymin) / dy)
* internal cells are half-open, [low, high), so a point exactly on an internal edge
  belongs to the cell on its right or above it;
* the domain's upper edges x = xmax and y = ymax are clamped into the last
  column/row (otherwise they would map to a non-existent cell Px or Py);
* NaN and out-of-domain points must be screened out beforehand
  (``records.valid_mask``);
* empty cells (ranks with no observations) are valid.

The decomposition is non-periodic, and distances are plain Euclidean.
"""
from dataclasses import dataclass

import numpy as np

from .geometry import angle_to_chord, points, wrap_lon


@dataclass(frozen=True)
class Domain:
    """Axis-aligned rectangular domain."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @property
    def lx(self):
        return self.xmax - self.xmin

    @property
    def ly(self):
        return self.ymax - self.ymin


@dataclass(frozen=True)
class Decomposition:
    """Regular Px x Py decomposition of a Domain into equal cells, one per rank.

    Interface used by the halo code (shared with SphereDecomposition): geometry,
    points, owner, cell_rect, rect_distance, candidate_ranks, push_mask.
    """
    domain: Domain
    px: int
    py: int
    geometry = "plane"          # class attribute, not a dataclass field

    def points(self, x, y):
        """Coordinates whose Euclidean distance is the model distance."""
        return points(x, y, "plane")

    def candidate_ranks(self, rank, h):
        """Ranks that may need some of rank's points in their halo of width h."""
        x0, x1, y0, y1 = self.cell_rect(rank)
        return self.ranks_intersecting(x0 - h, x1 + h, y0 - h, y1 + h)

    def push_mask(self, dest, x, y, h):
        """Points to push to dest: those inside dest's cell expanded by h (a rectangle,
        so a superset of the exact halo; the receiver filters exactly)."""
        a0, a1, b0, b1 = self.cell_rect(dest)
        return (x >= a0 - h) & (x <= a1 + h) & (y >= b0 - h) & (y <= b1 + h)

    @property
    def size(self):
        """Number of ranks/subdomains, P = Px * Py."""
        return self.px * self.py

    @property
    def dx(self):
        return self.domain.lx / self.px

    @property
    def dy(self):
        return self.domain.ly / self.py

    def cell_ij(self, x, y):
        """Column and row indices (i, j) of points (vectorised).

        Applies the [low, high) rule and clamps x = xmax / y = ymax into the last
        cell. The caller must pass only screened (finite, in-domain) points.
        """
        d = self.domain
        i = np.floor((np.asarray(x) - d.xmin) / self.dx).astype(np.int64)
        j = np.floor((np.asarray(y) - d.ymin) / self.dy).astype(np.int64)
        return np.minimum(i, self.px - 1), np.minimum(j, self.py - 1)

    def owner(self, x, y):
        """Owning rank of points, i + Px * j (vectorised)."""
        i, j = self.cell_ij(x, y)
        return i + self.px * j

    def rank_ij(self, rank):
        """Inverse of the row-major numbering: rank -> (i, j)."""
        return rank % self.px, rank // self.px

    def cell_rect(self, rank):
        """Bounds (x0, x1, y0, y1) of the rank's core cell."""
        i, j = self.rank_ij(rank)
        d = self.domain
        return (d.xmin + i * self.dx, d.xmin + (i + 1) * self.dx,
                d.ymin + j * self.dy, d.ymin + (j + 1) * self.dy)

    def ranks_intersecting(self, x0, x1, y0, y1):
        """Ranks whose closed cell touches the closed rectangle [x0, x1] x [y0, y1].

        Used in halo discovery to limit which neighbours a rank talks to: only ranks
        whose cell meets "my cell expanded by h" can need my observations.
        """
        d = self.domain
        i0 = max(int(np.floor((x0 - d.xmin) / self.dx)), 0)
        i1 = min(int(np.floor((x1 - d.xmin) / self.dx)), self.px - 1)
        j0 = max(int(np.floor((y0 - d.ymin) / self.dy)), 0)
        j1 = min(int(np.floor((y1 - d.ymin) / self.dy)), self.py - 1)
        return np.array([i + self.px * j for j in range(j0, j1 + 1) for i in range(i0, i1 + 1)],
                        dtype=np.int64)

    def rect_distance(self, rank, x, y):
        """Euclidean distance from points to the rank's core cell (0 for points inside).

        This is the exact halo test used by the default "rect" criterion: a point is
        in the halo of `rank` if rect_distance <= h.
        """
        x0, x1, y0, y1 = self.cell_rect(rank)
        ddx = np.maximum(np.maximum(x0 - x, x - x1), 0.0)
        ddy = np.maximum(np.maximum(y0 - y, y - y1), 0.0)
        return np.hypot(ddx, ddy)


@dataclass(frozen=True)
class SphereDecomposition:
    """Equal-area Px x Py decomposition of the sphere: Py latitude bands x Px sectors.

    Points are (x, y) = (longitude, latitude) in radians (see geometry.py). Band j covers
    sin(lat) in [-1 + 2j/Py, -1 + 2(j+1)/Py), which makes every band, and every cell,
    of equal area. Sector i covers wrapped longitude [-pi + 2 pi i/Px, ...). Ranks are
    numbered as on the plane, owner = i + Px * j, with the same [low, high) rule and a
    clamp of lat = +pi/2 into the last band. Longitude is wrapped first, so 180 E and
    180 W are the same meridian and lon = 2 pi is lon = 0. A pole belongs to the sector
    of its (arbitrary) longitude; every point still has exactly one owner.

    The polar bands are wedges meeting at the pole. Distances (rect_distance) are
    exact chords, so the geometric halo is exact there too, only larger than for
    compact cells. Equal-regions partitions with polar caps are a later refinement.
    """
    px: int
    py: int
    geometry = "sphere"
    # For records.valid_mask: any finite longitude, latitude in [-pi/2, pi/2].
    domain = Domain(-np.inf, np.inf, -np.pi / 2, np.pi / 2)

    @property
    def size(self):
        return self.px * self.py

    def points(self, x, y):
        return points(x, y, "sphere")

    def rank_ij(self, rank):
        return rank % self.px, rank // self.px

    def owner(self, x, y):
        """Owning rank of points (vectorised)."""
        lon = wrap_lon(x)
        i = np.floor((lon + np.pi) / (2.0 * np.pi / self.px)).astype(np.int64)
        j = np.floor((np.sin(np.asarray(y, dtype=float)) + 1.0) / (2.0 / self.py)).astype(np.int64)
        return np.minimum(i, self.px - 1) + self.px * np.minimum(j, self.py - 1)

    def cell_rect(self, rank):
        """(lon0, lon1, lat0, lat1) of the rank's cell, in radians, lon0 in [-pi, pi)."""
        i, j = self.rank_ij(rank)
        dlon = 2.0 * np.pi / self.px
        z0, z1 = -1.0 + 2.0 * j / self.py, -1.0 + 2.0 * (j + 1) / self.py
        return (-np.pi + i * dlon, -np.pi + (i + 1) * dlon,
                float(np.arcsin(np.clip(z0, -1, 1))), float(np.arcsin(np.clip(z1, -1, 1))))

    def rect_distance(self, rank, x, y):
        """Exact chord distance (km) from points to the rank's cell; 0 inside.

        Outside the cell the nearest point lies on its boundary: two parallels and two
        meridian arcs.
        * Parallel at lat_k: if the point's longitude is within the cell, the nearest
          point is straight along the meridian, at angle |lat - lat_k|. Otherwise it
          is a corner, which the meridian arcs include.
        * Meridian arc at lon_m, lat in [lat0, lat1]: cos(angle) = A sin(phi) + B cos(phi)
          = sqrt(A^2 + B^2) cos(phi - phi*), with A = sin(lat), B = cos(lat) cos(lon - lon_m)
          and phi* = atan2(A, B). On the arc the maximum is either at phi* (if phi* lies on
          the arc) or at one of the two ends. Clipping phi* alone is not enough: phi* can be
          circularly nearer the far end (e.g. B < 0 with an arc in one hemisphere). So the
          clipped phi* and both ends are evaluated, and all three are needed.
        Accuracy: arccos loses precision for tiny angles (about 1e-8 rad, 0.1 m), which is
        irrelevant at km scales.
        """
        lon0, lon1, lat0, lat1 = self.cell_rect(rank)
        lon = np.asarray(x, dtype=float)
        lat = np.asarray(y, dtype=float)
        in_lon = np.mod(lon - lon0, 2.0 * np.pi) <= (lon1 - lon0)
        in_lat = (lat >= lat0) & (lat <= lat1)
        best = np.full(lat.shape, np.inf)
        for lat_k in (lat0, lat1):
            best = np.minimum(best, np.where(in_lon, np.abs(lat - lat_k), np.inf))
        A = np.sin(lat)
        for lon_m in (lon0, lon1):
            B = np.cos(lat) * np.cos(lon - lon_m)
            for phi in (np.clip(np.arctan2(A, B), lat0, lat1), lat0, lat1):
                cosang = np.clip(A * np.sin(phi) + B * np.cos(phi), -1.0, 1.0)
                best = np.minimum(best, np.arccos(cosang))
        best = np.where(in_lon & in_lat, 0.0, best)        # also works for scalar input
        return angle_to_chord(best)

    def candidate_ranks(self, rank, h):
        """All ranks; push_mask is exact, so only real neighbours receive anything.

        This costs O(P * n_local) distance evaluations per rank, which is fine for the
        prototype. A production version would prefilter by latitude band and sector
        (a bound on the angular distance between cells).
        """
        return np.arange(self.size)

    def push_mask(self, dest, x, y, h):
        """Points within h of dest's cell (exact)."""
        return self.rect_distance(dest, x, y) <= h
