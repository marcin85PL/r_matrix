"""Correlated-R prototype: a core + halo approximation to R^-1 on mock MPI tasks.

See README.md in this directory for the design and the results. Quick start::

    from r_matrix import Config, run
    res = run(Config(), h=2.0)        # h in domain units; the default l = c = 0.5
    res.metrics["rel_l2"]             # error against the exact global solve
    z, _ = res.apply_Rinv(d_orig)     # apply R^-1 to any vector on the original ranks
"""
from .comm import MockComm
from .decomposition import Decomposition, Domain
from .pipeline import Config, RunResult, run, sweep

__all__ = ["Config", "Decomposition", "Domain", "MockComm", "RunResult", "run", "sweep"]
