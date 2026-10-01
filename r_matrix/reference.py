"""Global reference solution, "as if there were no domain decomposition".

All observations are gathered, the full dense R is built, and R z* = d is solved
exactly by Cholesky. This is feasible only for small problems (a few thousand
observations) and serves solely to validate the core + halo approximation
(the NUMERICAL gate of deck slide 8).
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .covariance import build_R


def global_solve(owned, c, alpha):
    """Exact R^-1 d over all observations.

    Parameters
    ----------
    owned : per-rank owned records (their "value" field is d)
    c, alpha : covariance parameters, as in covariance.build_R

    Returns
    -------
    gids : all global IDs, sorted
    z : exact R^-1 d, aligned with gids
    R : the global covariance matrix in the same order (used for the true residual)
    factor : its Cholesky factor, for further exact solves (RunResult.apply_Rinv_exact)
    """
    allobs = np.concatenate(owned)
    allobs = allobs[np.argsort(allobs["global_id"])]
    R = build_R(allobs["x"], allobs["y"], allobs["sigma"], c, alpha)
    factor = cho_factor(R, lower=True)
    z = cho_solve(factor, allobs["value"])
    return allobs["global_id"], z, R, factor
