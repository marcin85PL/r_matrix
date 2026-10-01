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
    """Regular Px x Py decomposition of a Domain into equal cells, one per rank."""
    domain: Domain
    px: int
    py: int

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
