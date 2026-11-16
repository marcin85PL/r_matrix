"""Helpers to evaluate the Vecchia prototype (vecchia.py) against core + halo on the same data.

- ``golden_data``: the observations of a golden case (export.CASES), without exporting it
- ``distribute``: owned records per patch for a partition (as after redistribution)
- ``build``: one distributed Vecchia operator (point or block) on a MockComm, with the matching
  serial W assembled from the per-patch rows
- ``core_halo``: the dense core + halo operator A~ (and its symmetrisation) for the same data
- ``costs``: flops, memory and communication of one setup and one application
"""
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import vecchia as V
from .comm import MockComm
from .covariance import LocalSolver, build_R
from .export import CASES
from .generate import make_obs
from .geometry import points
from .halo import discover_halos


def golden_data(name):
    """(obs, params, h_of_group, active) for golden case `name`, generated as export.export_case.

    params : {group: (c, alpha)}; h_of_group : {group: halo width of its class}.
    """
    s = CASES[name]
    classes = s["classes"]
    groups = s.get("n_groups", 1)
    obs = make_obs(s["n"], None, 1, seed=s["seed"], pattern=s["pattern"], departures="correlated",
                   c=classes[0]["c"], alpha=classes[0]["alpha"], sigma_range=(1.0, 1.0),
                   n_groups=groups, group_layout=s.get("group_layout", "random"),
                   geometry="sphere")[0]
    n = len(obs)
    cog = np.asarray(s.get("class_of_group", [0] * groups), dtype=np.int64)
    active = np.ones(n, dtype=bool)
    if s.get("inactive"):
        active = np.random.default_rng(3000 + s["seed"]).random(n) >= s["inactive"]
    params = {g: (classes[cog[g]]["c"], classes[cog[g]]["alpha"]) for g in range(groups)}
    hs = {g: classes[cog[g]]["h"] for g in range(groups)}
    return obs, params, hs, active


def distribute(obs, decomp):
    """Owned records of every patch, sorted by global ID (what redistribution produces)."""
    own = decomp.owner(obs["x"], obs["y"])
    out = []
    for p in range(decomp.size):
        o = obs[own == p]
        out.append(o[np.argsort(o["global_id"], kind="stable")])
    return out


def orderings(obs, owner, seq, how):
    """Global ordering key per datum (gid -> rank), per group: "maxmin" or "patch" (patches in seq,
    max-min inside each patch)."""
    out = {}
    for g in np.unique(obs["group"]):
        idx = np.flatnonzero(obs["group"] == g)
        xyz = points(obs["x"][idx], obs["y"][idx], "sphere")
        gid = obs["global_id"][idx]
        if how == "maxmin":
            order = V.maxmin_order(xyz, gid)[0]
        elif how == "patch":
            order = V.patch_maxmin_order(xyz, gid, owner[idx], seq)
        else:
            raise ValueError(how)
        out.update(zip(gid[order].tolist(), range(len(order))))
    return out


@dataclass
class Built:
    comm: MockComm
    factors: list
    maps: list
    vowned: list
    halos: list
    halo_owner: list
    info: dict
    gids: np.ndarray          # all global IDs, sorted: the row/column order of W()

    def W(self):
        """The global W (sparse, rows and columns in sorted global-ID order) from the patch rows."""
        n = len(self.gids)
        rows, cols, vals = [], [], []
        for f, v, hh in zip(self.factors, self.vowned, self.halos):
            lg = np.concatenate([v["global_id"], hh["global_id"]])
            co = f.W_loc.tocoo()
            rows.append(np.searchsorted(self.gids, v["global_id"][co.row]))
            cols.append(np.searchsorted(self.gids, lg[co.col]))
            vals.append(co.data)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))


def build(obs, decomp, variant, params, n_tasks=None, m=30, order="maxmin", seq="id", h=None,
          R0=None, coarse=0, cap=np.inf):
    """One distributed Vecchia setup on a MockComm with decomp.size patches.

    variant "point": m nearest earlier neighbours, ordering `order`, position-halo radius R0
    (default: 4 c of the first group), `coarse` data replicated; "block": the patches in sequence
    `seq` ("id" or "colour"), one-sided halo of width h.
    n_tasks: patches are mapped to tasks in contiguous blocks (communication is counted between
    tasks), as pipeline.patch_task_map.
    """
    from .pipeline import patch_task_map
    P = decomp.size
    task_of = None if n_tasks is None else patch_task_map(P, n_tasks, "block")[0]
    comm = MockComm(P, task_of)
    owned = distribute(obs, decomp)
    owner = decomp.owner(obs["x"], obs["y"])
    xyz = points(obs["x"], obs["y"], "sphere")
    hseq = h if h is not None else 4 * next(iter(params.values()))[0]
    pseq = V.patch_sequence(xyz, owner, P, hseq, seq)[0]
    gids = np.sort(obs["global_id"])
    if variant == "point":
        rk = orderings(obs, owner, pseq, order)
        vown = V.to_vrec(owned, rk)
        R0 = 4 * next(iter(params.values()))[0] if R0 is None else R0
        f, maps, halos, ho, info = V.setup_point(comm, decomp, vown, m, params, R0, coarse, cap)
        info["seq"] = pseq
        return Built(comm, f, maps, vown, halos, ho, info, gids)
    if variant == "block":
        vown = V.to_vrec(owned, {int(g): 0 for g in obs["global_id"]})
        f, maps, halos, ho, vown = V.setup_block(comm, decomp, vown, params, h, pseq)
        return Built(comm, f, maps, vown, halos, ho, dict(seq=pseq), gids)
    raise ValueError(variant)


def core_halo(obs, decomp, c, alpha, h):
    """Dense core + halo operator for one group's data: (A~, (A~ + A~^T)/2, per-patch (ncore, nhalo)).

    Rows and columns in sorted global-ID order. As in r_matrix.export._layout: owned data per patch,
    "obs" halo criterion, local Cholesky on core + halo.
    """
    P = decomp.size
    comm = MockComm(P)
    owned = distribute(obs, decomp)
    halos, _, _ = discover_halos(comm, decomp, owned, h, "obs", "kdtree")
    gids = np.sort(obs["global_id"])
    n = len(gids)
    A = np.zeros((n, n))
    sizes = []
    for o, hh in zip(owned, halos):
        s = LocalSolver(o, hh, c, alpha, "sphere")
        sizes.append((s.ncore, s.nhalo))
        if s.ncore == 0:
            continue
        rows = np.searchsorted(gids, o["global_id"])
        cols = np.searchsorted(gids, np.concatenate([o["global_id"], hh["global_id"]]))
        A[np.ix_(rows, cols)] = s.solve(np.eye(s.n)[:, :s.ncore]).T
    return A, 0.5 * (A + A.T), sizes


def exact_C(obs, c, alpha):
    """Dense C of one group, rows and columns in sorted global-ID order."""
    o = obs[np.argsort(obs["global_id"], kind="stable")]
    return build_R(o["x"], o["y"], np.ones(len(o)), c, alpha, geometry="sphere")


def costs(built):
    """Setup and apply costs of a Built operator.

    flops_setup_max/total : the factorisation (per patch: max, and the sum)
    flops_apply_max/total : one C~^-1 x = W^T W x (two products with the stored W)
    words_stored_max      : stored W entries plus their column indices, per patch (8-byte words)
    Communication (from the MockComm counters): bytes/messages of setup, of one application
    (halo + adjoint halo), and the peers of the busiest patch.
    """
    f = built.factors
    st = built.comm.stats
    fs = np.array([x.flops_setup for x in f])
    fa = np.array([2 * x.flops_apply for x in f])
    words = np.array([2 * x.W_loc.nnz + x.ncore + 1 for x in f])
    ncore = np.array([x.ncore for x in f])
    nhalo = np.array([x.nhalo for x in f])
    peers = np.array([len(m_.recv) for m_ in built.maps])
    return dict(flops_setup_max=float(fs.max()), flops_setup_total=float(fs.sum()),
                flops_apply_max=float(fa.max()), flops_apply_total=float(fa.sum()),
                words_stored_max=int(words.max()), halo_over_core=float(nhalo.sum() / ncore.sum()),
                halo_max=int(nhalo.max()), peers_max=int(peers.max()),
                bytes_setup=st.total("setup")["bytes"], messages_setup=st.total("setup")["messages"],
                rounds_setup=built.info.get("rounds", 1))


def core_halo_costs(sizes, ngroups=1):
    """Flops of core + halo for per-patch (ncore, nhalo) block sizes: setup n^3 / 3, symmetrised
    apply 2 right-hand sides x 2 triangular solves = 4 n^2; stored factor n^2 / 2 words."""
    n = np.array([a + b for a, b in sizes], dtype=float)
    return dict(flops_setup_max=float((n ** 3 / 3).max()), flops_setup_total=float((n ** 3 / 3).sum()),
                flops_apply_max=float((4 * n ** 2).max()), flops_apply_total=float((4 * n ** 2).sum()),
                words_stored_max=int((n ** 2 / 2).max()),
                halo_over_core=float(sum(b for _, b in sizes) / sum(a for a, _ in sizes)))


__all__ = ["Built", "build", "core_halo", "core_halo_costs", "costs", "distribute", "exact_C",
           "golden_data", "orderings"]
