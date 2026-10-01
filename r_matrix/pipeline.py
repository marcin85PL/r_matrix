"""End-to-end prototype driver (deck slides 1 and 8).

``run(cfg)`` executes the whole sequence on mock MPI tasks and validates it:

    generate            random initial ownership                 generate.make_obs
    --- setup, once per observation geometry ------------------------------------------
    1 spatial owner     owner = i + Px j                         decomposition
    2 redistribute      Alltoall counts + Alltoallv records      redistribute.redistribute
    3 halo discovery    geometric push + exact filter            halo.discover_halos
    4 freeze comm map   request / translate / confirm            commmap.build_comm_maps
    5a factorise        dense Cholesky of R on core + halo       covariance.LocalSolver
    --- recurring, at every application of R^-1 (RunResult.apply_Rinv) -----------------
    forward             d: original ranks -> spatial owners      RedistributionPlan.forward
    halo update         d on the halo from neighbours            commmap.halo_exchange
    5b local solve      z_loc = R_loc^-1 d_loc, keep the core    LocalSolver.apply
    reverse             z: spatial owners -> original ranks      RedistributionPlan.reverse
    --- validation ----------------------------------------------------------------------
    distributed residual  ||R z~ - d|| / ||d|| via one more halo exchange
    global reference      exact z* = R^-1 d by one global Cholesky  reference.global_solve

The error of the approximation is z~ - z*, compared on the original ranks. That
comparison also validates the reverse trip. All metrics are collected in
``RunResult.metrics``; see ``run`` for the list.
"""
import time
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
from scipy.linalg import cho_solve

from .comm import MockComm
from .commmap import build_comm_maps, halo_exchange
from .covariance import LocalSolver
from .decomposition import Decomposition, Domain
from .diagnostics import error_metrics, halo_summary, load_balance
from .generate import make_obs
from .halo import discover_halos
from .redistribute import check_redistribution, redistribute
from .reference import global_solve


@dataclass(frozen=True)
class Config:
    """All knobs of one experiment. The defaults are the baseline used in the notebook.

    Geometry and decomposition
        n_obs, domain, px, py : observations, the domain, and Px x Py tasks. The
            default is 3000 obs on 12 x 9, with 4 x 3 tasks and 3 x 3 cells, about
            7 obs per l^2.
    Observation generator (generate.make_obs)
        seed, pattern, departures, empty_init_ranks, skew
    Covariance (covariance.build_R)
        c : Gaspari-Cohn length scale l; correlations vanish beyond 2c
        alpha : correlated fraction of the error variance (1 - alpha is the nugget)
    Halo (halo.discover_halos)
        h : halo width, in domain units (not in units of l). Results are quoted
            as h / l, so with the default c = 0.5, run(h=2.0) means h = 4 l.
        criterion : "rect" or "obs"; halo_method : "brute" or "kdtree"
    reference : also run the global exact solve (needed for the error metrics)
    """
    n_obs: int = 3000
    domain: Domain = field(default_factory=lambda: Domain(0.0, 12.0, 0.0, 9.0))
    px: int = 4
    py: int = 3
    seed: int = 0
    pattern: str = "uniform"
    departures: str = "correlated"
    empty_init_ranks: tuple = ()
    skew: Optional[float] = None
    c: float = 0.5
    alpha: float = 0.8
    h: float = 1.0
    criterion: str = "rect"
    halo_method: str = "brute"
    reference: bool = True

    @property
    def h_over_l(self):
        return self.h / self.c


@dataclass
class RunResult:
    """Everything a run produced. Lists are indexed by rank.

    obs_init   : records on the initial (random) ranks
    owned      : records after redistribution, sorted by global_id
    halos, halo_owner, halo_stats : from halo discovery
    maps, plan, solvers : the frozen setup, reused by apply_Rinv
    z_owned    : approximate R^-1 d on the spatial owners (owned order)
    z_orig     : the same values back on the original ranks
    z_ref_orig : exact R^-1 d aligned with z_orig (NaN if reference=False)
    comm       : the MockComm; comm.stats holds the per-phase traffic
    timings    : wall-clock seconds per stage (single process, so indicative only)
    metrics    : a flat dict of scalar diagnostics (see run)
    """
    config: Config
    decomp: Decomposition
    obs_init: list
    owned: list
    halos: list
    halo_owner: list
    halo_stats: list
    maps: list
    plan: object
    solvers: list
    z_owned: list
    z_orig: list
    z_ref_orig: list
    comm: MockComm
    timings: dict
    metrics: dict
    ref_gids: Optional[np.ndarray] = None     # sorted global IDs of the reference solve
    ref_factor: Optional[tuple] = None        # Cholesky factor of the global R

    def apply_Rinv(self, d_orig, tag="recurring"):
        """Approximate R^-1 d for a vector living on the original ranks.

        This is the recurring path: forward -> halo exchange -> local solves ->
        reverse, using only the frozen setup. d_orig[r] has one entry per record
        initially on rank r. It returns (z_orig, z_owned).
        """
        d_core = self.plan.forward(self.comm, d_orig, phase=f"{tag}:forward")
        d_halo = halo_exchange(self.comm, self.maps, d_core, phase=f"{tag}:halo")
        z_owned = [s.apply(dc, dh) for s, dc, dh in zip(self.solvers, d_core, d_halo)]
        return self.plan.reverse(self.comm, z_owned, phase=f"{tag}:reverse"), z_owned

    def apply_Rinv_exact(self, d_orig):
        """Exact R^-1 d via the global factor, laid out like d_orig (needs reference=True)."""
        gids = np.concatenate([o["global_id"] for o in self.obs_init])
        d = np.concatenate(d_orig)
        ok = np.isin(gids, self.ref_gids)                      # skip rejected records
        pos = np.searchsorted(self.ref_gids, gids[ok])
        d_sorted = np.zeros(len(self.ref_gids))
        d_sorted[pos] = d[ok]
        z_all = np.full(len(gids), np.nan)
        z_all[ok] = cho_solve(self.ref_factor, d_sorted)[pos]
        return np.split(z_all, np.cumsum([len(o) for o in self.obs_init])[:-1])

    def owned_errors(self):
        """Per rank, the error z~ - z* in owned order; for maps of where the error lives."""
        gids = np.concatenate([o["global_id"] for o in self.obs_init])
        zref = np.concatenate(self.z_ref_orig)
        order = np.argsort(gids)
        out = []
        for o, z in zip(self.owned, self.z_owned):
            out.append(z - zref[order][np.searchsorted(gids[order], o["global_id"])])
        return out


def run(cfg=Config(), **overrides):
    """Run the prototype once. Keyword overrides replace Config fields, e.g. run(h=2.0).

    Metrics (RunResult.metrics)
    ---------------------------
    Accuracy (only when cfg.reference):
        rel_l2, max_abs, rel_max : z~ against the exact z*
        residual_global : ||R z~ - d|| / ||d|| with the global R
    Accuracy (always):
        residual_distributed : the same residual from the halo matvec; it equals
            residual_global when matvec_exact (h >= 2c)
    Load and halo:
        owned : {min, mean, max, cv} of owned counts
        halo_total, candidates_total, halo_core_ratio_mean/peak, local_n_max
        local_R_bytes_max, map_bytes_total, halo_peers_max
    Communication, in bytes counted by MockComm:
        bytes_setup : everything tagged "setup:*"
        bytes_apply : one application of R^-1 (forward + halo + reverse)
        bytes_apply_halo : the halo-update part of bytes_apply alone
        messages_apply : non-empty off-rank messages in one application
    Time (single process, indicative): time_setup, time_apply, time_reference
    """
    cfg = replace(cfg, **overrides)
    decomp = Decomposition(cfg.domain, cfg.px, cfg.py)
    P = decomp.size
    comm = MockComm(P)
    t = {}

    obs_init = make_obs(cfg.n_obs, cfg.domain, P, seed=cfg.seed, pattern=cfg.pattern,
                        decomposition=decomp, departures=cfg.departures, c=cfg.c,
                        alpha=cfg.alpha, empty_ranks=cfg.empty_init_ranks, skew=cfg.skew)

    # ---- setup (once per geometry) --------------------------------------------------
    t0 = time.perf_counter()
    owned, rejected, plan = redistribute(comm, decomp, obs_init)
    t["setup_redistribute"] = time.perf_counter() - t0
    check_redistribution(obs_init, owned, rejected, decomp)

    t0 = time.perf_counter()
    halos, halo_owner, hstats = discover_halos(comm, decomp, owned, cfg.h, cfg.criterion,
                                               cfg.halo_method)
    t["setup_halo"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    maps = build_comm_maps(comm, owned, halos, halo_owner)
    t["setup_map"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    solvers = [LocalSolver(o, hh, cfg.c, cfg.alpha) for o, hh in zip(owned, halos)]
    t["setup_factorise"] = time.perf_counter() - t0

    # ---- recurring: apply R^-1 to the departures living on the original ranks ---------
    # (The same steps as RunResult.apply_Rinv, inlined so that each stage is timed.)
    d_orig = [o["value"].copy() for o in obs_init]
    t0 = time.perf_counter()
    d_core = plan.forward(comm, d_orig)
    d_halo = halo_exchange(comm, maps, d_core)
    t["apply_comm_in"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_owned = [s.apply(dc, dh) for s, dc, dh in zip(solvers, d_core, d_halo)]
    t["apply_solve"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    z_orig = plan.reverse(comm, z_owned)
    t["apply_comm_out"] = time.perf_counter() - t0

    # ---- distributed residual ||R z~ - d|| / ||d|| (exact matvec when h >= 2c) --------
    # Tagged "diag" so that it does not count as recurring cost.
    z_halo = halo_exchange(comm, maps, z_owned, phase="diag:halo")
    res = np.concatenate([s.matvec_core(zc, zh) - dc
                          for s, zc, zh, dc in zip(solvers, z_owned, z_halo, d_core)])
    d_all = np.concatenate(d_core)

    stats = comm.stats
    m = {
        "h_over_l": cfg.h_over_l,
        "n_valid": int(d_all.size),
        "n_rejected": int(sum(len(r) for r in rejected)),
        "residual_distributed": float(np.linalg.norm(res) / np.linalg.norm(d_all)),
        "matvec_exact": bool(cfg.h >= 2 * cfg.c),
        "owned": load_balance([len(o) for o in owned]),
        **halo_summary(hstats),
        "local_R_bytes_max": int(max(s.nbytes for s in solvers)),
        "map_bytes_total": int(sum(mp.nbytes for mp in maps)),
        "halo_peers_max": int(max(len(mp.recv) for mp in maps)),
        "bytes_setup": stats.total("setup")["bytes"],
        "bytes_apply": stats.total("recurring")["bytes"],
        "bytes_apply_halo": stats.total("recurring:halo")["bytes"],
        "messages_apply": stats.total("recurring")["messages"],
        "time_setup": sum(v for k, v in t.items() if k.startswith("setup")),
        "time_apply": sum(v for k, v in t.items() if k.startswith("apply")),
    }

    # ---- global reference "as if there were no domain decomposition" -----------------
    z_ref_orig = [np.full(len(o), np.nan) for o in obs_init]
    ref_gids = ref_factor = None
    if cfg.reference:
        t0 = time.perf_counter()
        ref_gids, zref, R, ref_factor = global_solve(owned, cfg.c, cfg.alpha)
        t["reference"] = time.perf_counter() - t0
        # Align z* with the original layout, and compare where z~ came back.
        for r, o in enumerate(obs_init):
            idx = np.searchsorted(ref_gids, o["global_id"])
            ok = np.isfinite(z_orig[r])
            z_ref_orig[r][ok] = zref[idx[ok]]
        z = np.concatenate(z_orig)
        zr = np.concatenate(z_ref_orig)
        ok = np.isfinite(z)
        m.update(error_metrics(z[ok], zr[ok]))
        # True global residual, independent of the halo matvec.
        order = np.argsort(np.concatenate([o["global_id"] for o in owned]))
        z_sorted, d_sorted = np.concatenate(z_owned)[order], d_all[order]
        m["residual_global"] = float(np.linalg.norm(R @ z_sorted - d_sorted) / np.linalg.norm(d_sorted))
        m["time_reference"] = t["reference"]

    return RunResult(cfg, decomp, obs_init, owned, halos, halo_owner, hstats, maps, plan,
                     solvers, z_owned, z_orig, z_ref_orig, comm, t, m, ref_gids, ref_factor)


def sweep(param, values, cfg=Config(), **fixed):
    """Run once per value of one Config field (with the other overrides in `fixed`).

    Returns a list of flat metric dicts (ready for pandas.DataFrame). The nested
    "owned" load-balance dict is flattened into owned_min, owned_mean, and so on.
    """
    rows = []
    for v in values:
        res = run(cfg, **{**fixed, param: v})
        row = {param: v, **{k: val for k, val in res.metrics.items() if not isinstance(val, dict)}}
        row.update({f"owned_{k}": val for k, val in res.metrics["owned"].items()})
        rows.append(row)
    return rows
