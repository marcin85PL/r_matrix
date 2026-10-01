"""Observation-error covariance model and the local core + halo solver (deck slides 1 and 8).

Covariance model
----------------
    R_ij = sigma_i * sigma_j * [ alpha * rho(|x_i - x_j|) + (1 - alpha) * delta_ij ]

* rho is the Gaspari-Cohn (1999) compactly supported correlation with length scale
  c, which is the "l" used throughout (h/l means h/c). rho(0) = 1, and rho(r) = 0
  exactly for r >= 2c, so R is sparse. GC is positive definite in 2-D and 3-D.
* alpha in (0, 1] is the correlated fraction of the error variance. The remaining
  1 - alpha is an uncorrelated "nugget". It keeps R well-conditioned, including for
  coincident observations, which would otherwise give identical rows. It also
  controls how fast the entries of R^-1 decay with distance: the smaller the
  nugget, the slower the decay and the wider the halo needed (notebook section 3).

Why a halo, and why this is an approximation
--------------------------------------------
R is sparse, but R^-1 is dense, with entries that decay roughly exponentially with
distance. Each rank therefore solves the *local* system R_loc z_loc = d_loc over
its core plus halo observations and keeps only the core part of z_loc. Core
observations near the edge of the halo "see" an artificially truncated problem.
That truncation is the approximation error, and it falls as the halo width h grows.

Products with R itself (``LocalSolver.matvec_core``) are exact whenever h >= 2c,
because every nonzero R_ij with i in the core then has j in core + halo.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist


def gaspari_cohn(r, c):
    """Gaspari & Cohn (1999, eq. 4.10) fifth-order piecewise rational correlation.

    Parameters
    ----------
    r : array of distances
    c : length scale; the support is 2c

    Returns
    -------
    rho(r) with the same shape as r: 1 at r = 0, smooth (C^2), and 0 for r >= 2c.
    """
    z = np.abs(np.asarray(r, dtype=float)) / c
    out = np.zeros_like(z)
    m1 = z <= 1.0
    m2 = (z > 1.0) & (z < 2.0)
    a = z[m1]
    out[m1] = -0.25 * a**5 + 0.5 * a**4 + 0.625 * a**3 - 5.0 / 3.0 * a**2 + 1.0
    b = z[m2]
    out[m2] = (b**5 / 12.0 - 0.5 * b**4 + 0.625 * b**3 + 5.0 / 3.0 * b**2
               - 5.0 * b + 4.0 - 2.0 / (3.0 * b))
    return out


def build_R(x, y, sigma, c, alpha):
    """Dense n x n covariance R for points (x, y) with error std sigma (see module doc).

    Dense storage is deliberate: the prototype targets correctness on small problems.
    A production version would exploit the sparsity of R.
    """
    xy = np.column_stack([x, y])
    C = alpha * gaspari_cohn(cdist(xy, xy), c)
    C[np.diag_indices_from(C)] += 1.0 - alpha
    return sigma[:, None] * C * sigma[None, :]


class LocalSolver:
    """Per-rank dense Cholesky solver on core + halo.

    The local ordering is [core observations (owned, sorted by global_id),
    halo observations (in comm-map order)], so the first ``ncore`` entries of every
    local vector are the core.

    Setup (once per geometry): build R_loc and factorise it, R_loc = L L^T.
    Apply (every iteration): solve R_loc z = [d_core; d_halo] and return z[:ncore].
    Solving on the halo too, then discarding it, is the core + halo approximation.
    """

    def __init__(self, core, halo, c, alpha):
        """core, halo : OBS_DTYPE arrays (only x, y and sigma are used)."""
        self.ncore = len(core)
        self.nhalo = len(halo)
        loc = np.concatenate([core, halo])
        self.R = build_R(loc["x"], loc["y"], loc["sigma"], c, alpha)
        self.factor = cho_factor(self.R, lower=True) if len(loc) else None

    @property
    def n(self):
        """Local problem size, ncore + nhalo."""
        return self.ncore + self.nhalo

    def apply(self, d_core, d_halo):
        """Approximate (R^-1 d) on the core. An empty rank returns an empty array."""
        if self.ncore == 0:
            return np.zeros(0)
        z = cho_solve(self.factor, np.concatenate([d_core, d_halo]))
        return z[:self.ncore]

    def matvec_core(self, z_core, z_halo):
        """(R z) restricted to the core, from core + halo values of z.

        Exact whenever the halo covers the correlation support (h >= 2c). The
        pipeline uses it to compute the residual R z~ - d without a global matrix.
        """
        if self.ncore == 0:
            return np.zeros(0)
        return self.R[:self.ncore] @ np.concatenate([z_core, z_halo])

    @property
    def nbytes(self):
        """Memory of the stored local R (the Cholesky factor is the same size again)."""
        return self.R.nbytes
