"""Halo discovery (deck slide 5).

After redistribution, each rank owns the observations in its core cell. To solve
the local core + halo problem, it also needs copies of nearby observations owned
by other ranks, which form its *halo*. Discovery runs once per geometry, in two
steps:

1. Geometric candidate acquisition, a push from the owner. Every rank s sends each
   owned record to every other rank d whose core cell, expanded by h (a rectangle),
   contains the record. Only ranks near s's own cell can qualify, so s talks to few
   peers. The result is a superset of the true halo.

2. Exact filter at the receiver. Rank d keeps the candidates that satisfy the halo
   criterion, deduplicates them by global_id, and sorts them by
   (owner_rank, global_id). That ordering makes the halo array directly usable by
   the communication map: the halo from each owner is one contiguous, ID-sorted
   slice.

Halo criteria (``criterion``)
-----------------------------
"rect"  distance(point, receiver's core rectangle) <= h. This is the default. It
        depends only on geometry, not on where observations happen to be, and with
        h >= 2c it guarantees that every observation correlated with a core
        observation is in the halo.
"obs"   distance(point, nearest *owned observation*) <= h. It is adaptive and gives
        a smaller halo (about 7 % smaller in the default setup), with the same
        guarantee for correlations of core observations.

Filter implementations (``method``, used by the "obs" criterion only)
----------------------------------------------------------------------
"brute"   direct pairwise distances; the reference
"kdtree"  a scipy cKDTree built on the candidates and queried with a ball of
          radius h around each owned observation

The deck's decision rule requires both methods to return identical halo ID sets
(tested). For "rect" the exact test is already O(n) per rank, so a tree brings
nothing and ``method`` is ignored.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class HaloStats:
    """Per-rank discovery diagnostics."""
    candidates: int   # records received in the geometric push (before filtering)
    halo: int         # unique records kept
    core: int         # owned records


def _push_candidates(comm, decomp, owned, h):
    """Step 1: each owner pushes records to ranks whose expanded core contains them.

    Returns (candidates, owners): per receiving rank, the received records and the
    source rank of each one (packed by source, as alltoallv delivers them).
    """
    P = comm.size
    sendbufs, sendcounts = [], []
    for s, o in enumerate(owned):
        x0, x1, y0, y1 = decomp.cell_rect(s)
        parts, counts = [], np.zeros(P, dtype=np.int64)
        # A rank whose expanded cell contains one of my points must have a cell
        # within h of my cell, so only those ranks are examined.
        for d in decomp.ranks_intersecting(x0 - h, x1 + h, y0 - h, y1 + h):
            if d == s:
                continue
            a0, a1, b0, b1 = decomp.cell_rect(d)
            m = (o["x"] >= a0 - h) & (o["x"] <= a1 + h) & (o["y"] >= b0 - h) & (o["y"] <= b1 + h)
            parts.append((d, o[m]))
            counts[d] = m.sum()
        parts.sort(key=lambda t: t[0])                    # pack by destination rank
        sendbufs.append(np.concatenate([p for _, p in parts]) if parts else o[:0])
        sendcounts.append(counts)
    comm.alltoall(sendcounts, "setup:halo:counts")
    recvbufs, recvcounts = comm.alltoallv(sendbufs, sendcounts, "setup:halo:records")
    owners = [np.repeat(np.arange(P), rc) for rc in recvcounts]
    return recvbufs, owners


def _within_obs_brute(cand, core, h, chunk=2048):
    """Mask of candidates within distance h of at least one core observation."""
    keep = np.zeros(len(cand), dtype=bool)
    if len(core) == 0:
        return keep
    cxy = np.column_stack([core["x"], core["y"]])
    for i in range(0, len(cand), chunk):                  # chunked to bound memory
        p = np.column_stack([cand["x"][i:i + chunk], cand["y"][i:i + chunk]])
        d2 = ((p[:, None, :] - cxy[None, :, :]) ** 2).sum(-1)
        keep[i:i + chunk] = np.sqrt(d2.min(axis=1)) <= h
    return keep


def _within_obs_kdtree(cand, core, h):
    """The same mask as _within_obs_brute, using a k-d tree on the candidates."""
    keep = np.zeros(len(cand), dtype=bool)
    if len(core) == 0 or len(cand) == 0:
        return keep
    tree = cKDTree(np.column_stack([cand["x"], cand["y"]]))
    for hits in tree.query_ball_point(np.column_stack([core["x"], core["y"]]), r=h):
        keep[hits] = True
    return keep


def discover_halos(comm, decomp, owned, h, criterion="rect", method="brute"):
    """Find each rank's halo (see the module docstring).

    Parameters
    ----------
    comm : comm.MockComm
    decomp : decomposition.Decomposition
    owned : per-rank owned records, from redistribute.redistribute
    h : halo width, in the same units as x, y
    criterion : "rect" or "obs"
    method : "brute" or "kdtree" (for criterion "obs")

    Returns
    -------
    halos : per rank, OBS_DTYPE copies of the halo records, sorted by
        (owner_rank, global_id); the position is halo_local_index
    halo_owner : per rank, an int array giving the owning rank of each halo record
    stats : per rank, HaloStats
    """
    cands, owners = _push_candidates(comm, decomp, owned, h)
    halos, halo_owner, stats = [], [], []
    for r, (cand, own) in enumerate(zip(cands, owners)):
        if criterion == "rect":
            keep = decomp.rect_distance(r, cand["x"], cand["y"]) <= h
        elif criterion == "obs":
            keep = (_within_obs_kdtree if method == "kdtree" else _within_obs_brute)(cand, owned[r], h)
        else:
            raise ValueError(f"unknown criterion {criterion!r}")
        c, o = cand[keep], own[keep]
        # Deduplicate by global ID. Each owner pushes a record at most once per
        # receiver, so this is a no-op here, but it is part of the contract for other
        # acquisition schemes.
        _, first = np.unique(c["global_id"], return_index=True)
        c, o = c[first], o[first]
        order = np.lexsort((c["global_id"], o))           # sort by owner, then by ID
        halos.append(c[order])
        halo_owner.append(o[order])
        stats.append(HaloStats(candidates=len(cand), halo=len(c), core=len(owned[r])))
    return halos, halo_owner, stats
