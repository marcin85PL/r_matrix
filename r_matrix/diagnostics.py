"""Error metrics, cost / load-balance summaries, and dense operator matrices for analysis."""
import numpy as np
from scipy.linalg import cho_solve


def error_metrics(z, z_ref):
    """Compare the approximate z = R_approx^-1 d with the exact z_ref = R^-1 d.

    Returns
    -------
    rel_l2  : ||z - z_ref||_2 / ||z_ref||_2
    max_abs : max_i |z_i - z_ref_i|
    rel_max : max_abs / max_i |z_ref_i|
    """
    e = z - z_ref
    return {
        "rel_l2": float(np.linalg.norm(e) / np.linalg.norm(z_ref)),
        "max_abs": float(np.abs(e).max(initial=0.0)),
        "rel_max": float(np.abs(e).max(initial=0.0) / np.abs(z_ref).max()),
    }


def approx_matrix(res):
    """Assemble the approximate R^-1 (A~, before any symmetrisation) as a dense matrix.

    It is built directly from the local factors, without applying the operator:
    row block "core of rank p" = rows ncore of R_loc,p^-1, in the columns of p's
    [core; halo] observations. Rows and columns follow the sorted global IDs
    (res.ref_gids when the reference was run). The cost is one multi-RHS solve per rank,
    so it is fine for a few thousand observations. The symmetrised operator is
    (A + A.T) / 2.
    """
    gids = np.sort(np.concatenate([o["global_id"] for o in res.owned]))
    A = np.zeros((gids.size, gids.size))
    for o, hh, s in zip(res.owned, res.halos, res.solvers):
        if s.ncore == 0:
            continue
        rows = np.searchsorted(gids, o["global_id"])
        cols = np.searchsorted(gids, np.concatenate([o["global_id"], hh["global_id"]]))
        # Rows ncore of the symmetric R_loc^-1 = transpose of its first ncore columns.
        A[np.ix_(rows, cols)] = cho_solve(s.factor, np.eye(s.n)[:, :s.ncore]).T
    return A


def operator_matrix(apply, obs_init):
    """Dense matrix of a linear operator acting on the original layout (black box).

    apply : function taking a per-rank list of vectors and returning a per-rank list of
        results, e.g. ``lambda d: res.apply_Rinv(d)[0]``
    obs_init : per-rank records (only global_id and the per-rank lengths are used)

    It applies the operator to one unit vector per observation, so keep n to a few
    hundred. Rows and columns follow the sorted global IDs. Rejected observations, whose
    results come back as NaN, are dropped.
    """
    gids = np.concatenate([o["global_id"] for o in obs_init])
    splits = np.cumsum([len(o) for o in obs_init])[:-1]
    order = np.argsort(gids)
    cols = []
    for j in order:
        e = np.zeros(gids.size)
        e[j] = 1.0
        cols.append(np.concatenate(apply(np.split(e, splits)))[order])
    A = np.column_stack(cols)
    keep = ~np.isnan(A).any(axis=1)
    return A[np.ix_(keep, keep)]


def load_balance(counts):
    """min / mean / max of per-rank counts, and the coefficient of variation std/mean."""
    c = np.asarray(counts, dtype=float)
    mean = c.mean()
    return {"min": int(c.min()), "mean": float(mean), "max": int(c.max()),
            "cv": float(c.std() / mean) if mean > 0 else np.nan}


def halo_summary(halo_stats):
    """Aggregate the per-rank halo.HaloStats.

    halo_core_ratio_* is taken over non-empty ranks only. local_n_max is the largest
    local problem (core + halo), which drives the cost of the dense Cholesky:
    O(n^3) at setup and O(n^2) per apply.
    """
    ratio = np.array([s.halo / s.core for s in halo_stats if s.core > 0])
    return {
        "halo_total": int(sum(s.halo for s in halo_stats)),
        "candidates_total": int(sum(s.candidates for s in halo_stats)),
        "halo_core_ratio_mean": float(ratio.mean()) if ratio.size else np.nan,
        "halo_core_ratio_peak": float(ratio.max()) if ratio.size else np.nan,
        "local_n_max": int(max(s.core + s.halo for s in halo_stats)),
    }
