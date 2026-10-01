"""Redistribution from random to spatial ownership, and back (deck slide 4).

Setup, done once per observation geometry, by ``redistribute``
--------------------------------------------------------------
Each rank follows steps A-E of the deck:

A. Compute the destination: screen the records (``records.valid_mask``), then
   compute owner(x, y) for the survivors.
B. Count and pack: sort the records by destination rank and count them per
   destination.
C. Exchange counts: an all-to-all, so every rank learns how much it will receive.
D. Exchange records: an all-to-all-v of the full OBS_DTYPE records.
E. Re-index locally: sort the received records by global_id. The position in this
   sorted array is the ``owned_local_index`` used by everything downstream, and the
   ordering does not depend on arrival order.

The permutations of steps B and E are kept in a ``RedistributionPlan``.

Recurring, at every application of R^-1
---------------------------------------
Inside a minimisation, the vector that R^-1 acts on changes every iteration, but
the observation locations do not. ``RedistributionPlan.forward`` and
``RedistributionPlan.reverse`` therefore move bare float64 values using the frozen
counts and permutations. There is no re-screening, no owner computation and no
count exchange, and each value costs 8 bytes instead of a 48-byte record.
"""
from dataclasses import dataclass

import numpy as np

from .records import check_unique_ids, valid_mask


@dataclass
class RedistributionPlan:
    """Frozen routing between the original (random) layout and the spatial layout.

    Attributes (all lists are indexed by rank)
    ------------------------------------------
    n_orig : number of records initially on each rank, including rejected ones
    send_idx : per original rank, the indices of its valid records in send order
        (packed by destination)
    sendcounts : per original rank, the number of records sent to each destination
    recvcounts : per spatial rank, the number of records received from each source
    perm : per spatial rank, perm[k] is the position in the receive buffer of owned
        record k (the sort by global_id from step E)
    """
    n_orig: list
    send_idx: list
    sendcounts: list
    recvcounts: list
    perm: list

    def forward(self, comm, values, phase="recurring:forward"):
        """Original layout -> spatial layout.

        values[r] has length n_orig[r]; entries of rejected records are ignored.
        Returns, per rank, the values in owned (global_id-sorted) order.
        """
        bufs = [np.ascontiguousarray(v[idx], dtype=float) for v, idx in zip(values, self.send_idx)]
        recv, _ = comm.alltoallv(bufs, self.sendcounts, phase)
        return [rv[p] for rv, p in zip(recv, self.perm)]

    def reverse(self, comm, owned_values, phase="recurring:reverse"):
        """Spatial layout -> original layout (the exact inverse of forward).

        Returns, per original rank, an array of length n_orig[r]. The slots of
        records rejected at screening are NaN.
        """
        bufs = []
        for v, p in zip(owned_values, self.perm):
            b = np.empty(len(p))
            b[p] = v                          # undo step E: back to source-packed order
            bufs.append(b)
        recv, _ = comm.alltoallv(bufs, self.recvcounts, phase)   # counts swap roles
        out = []
        for n, idx, rv in zip(self.n_orig, self.send_idx, recv):
            z = np.full(n, np.nan)
            z[idx] = rv                       # undo step B: back to original positions
            out.append(z)
        return out


def redistribute(comm, decomp, obs_per_rank):
    """Move every valid record to its spatial owner (setup steps A-E).

    Parameters
    ----------
    comm : comm.MockComm with comm.size == decomp.size
    decomp : decomposition.Decomposition
    obs_per_rank : list of OBS_DTYPE arrays, one per initial rank

    Returns
    -------
    owned : list of OBS_DTYPE arrays; owned[r] holds the records inside cell r,
        sorted by global_id (the position is owned_local_index)
    rejected : list of OBS_DTYPE arrays; the records screened out on each initial rank
    plan : RedistributionPlan for recurring value transfers
    """
    P = comm.size
    if P != decomp.size:
        raise ValueError("communicator size must equal the number of subdomains")
    if len(obs_per_rank) != P:
        raise ValueError("expected one input array per rank")
    sendbufs, sendcounts, send_idx, rejected = [], [], [], []
    for obs in obs_per_rank:
        ok = valid_mask(obs, decomp.domain)                             # A: reject before communicating
        valid_idx = np.flatnonzero(ok)
        rejected.append(obs[~ok])
        valid = obs[valid_idx]
        dest = decomp.owner(valid["x"], valid["y"])
        order = np.argsort(dest, kind="stable")                         # B: pack by destination
        send_idx.append(valid_idx[order])
        sendbufs.append(valid[order])
        sendcounts.append(np.bincount(dest, minlength=P).astype(np.int64))
    recvcounts = comm.alltoall(sendcounts, "setup:redistribute:counts")  # C
    recvbufs, _ = comm.alltoallv(sendbufs, sendcounts, "setup:redistribute:records")  # D
    owned, perm = [], []
    for recv in recvbufs:                                               # E: re-index locally
        p = np.argsort(recv["global_id"], kind="stable")
        o = recv[p]
        check_unique_ids(o)
        owned.append(o)
        perm.append(p)
    plan = RedistributionPlan(n_orig=[len(o) for o in obs_per_rank], send_idx=send_idx,
                              sendcounts=sendcounts, recvcounts=recvcounts, perm=perm)
    return owned, rejected, plan


def check_redistribution(before, owned, rejected, decomp):
    """Assert the REDISTRIBUTION gate of deck slide 8. Raises AssertionError on failure.

    * The count is preserved: owned + rejected == input.
    * Global IDs are unique across all ranks (catches duplicates that landed on
      different ranks).
    * The ID set is preserved: nothing was lost or invented.
    * Every owned point lies inside its owner's cell.

    It gathers IDs from all ranks, so it is a diagnostic, not part of the algorithm.
    In MPI it would be an allreduce of counts plus a hash.
    """
    ids_in = np.concatenate([b["global_id"] for b in before])
    ids_rej = np.concatenate([r["global_id"] for r in rejected])
    ids_out = np.concatenate([o["global_id"] for o in owned])
    if ids_out.size + ids_rej.size != ids_in.size:
        raise AssertionError("record count not preserved")
    if np.unique(ids_out).size != ids_out.size:
        raise AssertionError("global IDs not unique after redistribution")
    if not np.array_equal(np.sort(np.concatenate([ids_out, ids_rej])), np.sort(ids_in)):
        raise AssertionError("ID set not preserved")
    for r, o in enumerate(owned):
        if len(o) and not np.all(decomp.owner(o["x"], o["y"]) == r):
            raise AssertionError(f"rank {r} holds a point outside its subdomain")
