"""Error metrics and cost / load-balance summaries used in RunResult.metrics."""
import numpy as np


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
