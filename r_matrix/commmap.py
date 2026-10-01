"""Frozen, deterministic halo communication map (deck slide 7).

Halo discovery is expensive and happens once. What recurs, at every application
of R^-1, is refreshing the *values* in the halo (the new d, or z for the residual
check). This module builds, once, a map that says exactly which owned values go
to which neighbour and in what order. After that, a halo update is a single
sparse exchange of float64 arrays with no searching.

Per-rank data structures (``RankMap``)
--------------------------------------
owned observations   global_id -> owned_local_index; implicit, because
                     owned[r] is sorted by global_id
halo observations    halos[r] sorted by (owner_rank, global_id); the position is
                     halo_local_index
receive map          recv[owner_rank] = slice of halos[r], sorted unique IDs
send map             send[dest_rank] = owned local indices, in exactly the ID order
                     the destination expects

Setup handshake (``build_comm_maps``)
-------------------------------------
1-2. Each rank sends each owner the sorted list of halo IDs it needs from that
     owner (an all-to-all of counts, then an all-to-all-v of int64 IDs).
3.   The owner validates that it really owns every requested ID, then translates
     the IDs into owned_local_index. The result is its send map.
4.   The owner returns (count, hash of the IDs in its packing order), and the
     requester checks it against its receive map. A mismatch in count, IDs or order
     is caught here, at setup, rather than as silently wrong numbers later.

The recurring exchange (``halo_exchange``) uses a sparse neighbour exchange, which
stands in for persistent point-to-point or MPI_Neighbor_alltoallv. Zero-count peers
are omitted.
"""
import hashlib
from dataclasses import dataclass, field

import numpy as np


def _id_hash(ids):
    """Order-sensitive 56-bit hash of an int64 ID sequence; fits in one int64."""
    return int.from_bytes(hashlib.blake2b(np.ascontiguousarray(ids, dtype=np.int64).tobytes(),
                                          digest_size=7).digest(), "little")


@dataclass
class RankMap:
    """Frozen communication map of one rank."""
    rank: int
    recv: dict = field(default_factory=dict)   # owner_rank -> slice into halos[rank]
    send: dict = field(default_factory=dict)   # dest_rank  -> owned local indices (int array)
    nhalo: int = 0

    @property
    def nbytes(self):
        """Approximate memory of the map: the send index arrays plus the recv slices."""
        return sum(v.nbytes for v in self.send.values()) + 16 * len(self.recv)


def build_comm_maps(comm, owned, halos, halo_owner):
    """Run the setup handshake and return one RankMap per rank.

    Parameters
    ----------
    owned : per-rank owned records (sorted by global_id)
    halos, halo_owner : from halo.discover_halos (sorted by (owner, global_id))

    Raises
    ------
    ValueError if a rank requests an ID that its owner does not hold.
    AssertionError if the count/hash confirmation disagrees.
    """
    maps = [RankMap(rank=r, nhalo=len(halos[r])) for r in range(comm.size)]
    reqs, reqcounts = _send_requests(comm, maps, halos, halo_owner)
    confs, confcounts = _translate_requests(comm, maps, owned, reqs, reqcounts)
    _check_confirmations(maps, halos, confs, confcounts)
    return maps


def _send_requests(comm, maps, halos, halo_owner):
    """Steps 1-2: build each receive map and send each owner the sorted IDs requested from it.

    halos[r] is already sorted by (owner, id), so the request buffer is the halo ID
    column as it stands, and the halo from each owner is one contiguous slice.
    """
    P = comm.size
    sendbufs, sendcounts = [], []
    for r in range(P):
        counts = np.bincount(halo_owner[r], minlength=P).astype(np.int64)
        start = np.concatenate([[0], np.cumsum(counts)])
        for s in np.flatnonzero(counts):
            maps[r].recv[int(s)] = slice(int(start[s]), int(start[s + 1]))
        sendbufs.append(halos[r]["global_id"].copy())
        sendcounts.append(counts)
    comm.alltoall(sendcounts, "setup:map:counts")
    return comm.alltoallv(sendbufs, sendcounts, "setup:map:requests")


def _translate_requests(comm, maps, owned, reqs, reqcounts):
    """Step 3: the owner validates the requested IDs and translates them into owned_local_index.

    It uses a binary search in its global_id-sorted owned array, which gives the send
    map. It then returns (count, hash) to each requester (step 4, send side).
    """
    P = comm.size
    confirm_bufs, confirm_counts = [], []
    for s in range(P):
        gids = owned[s]["global_id"]
        start = np.concatenate([[0], np.cumsum(reqcounts[s])])
        conf, counts = [], np.zeros(P, dtype=np.int64)
        for r in np.flatnonzero(reqcounts[s]):
            ids = reqs[s][start[r]:start[r + 1]]
            idx = np.searchsorted(gids, ids)
            if np.any(idx >= len(gids)) or np.any(gids[np.minimum(idx, len(gids) - 1)] != ids):
                raise ValueError(f"rank {s}: rank {r} requested IDs it does not own")
            maps[s].send[int(r)] = idx
            conf.append([len(ids), _id_hash(gids[idx])])
            counts[r] = 2                              # two int64s per requester
        confirm_bufs.append(np.array(conf, dtype=np.int64).ravel())
        confirm_counts.append(counts)
    comm.alltoall(confirm_counts, "setup:map:confirm_counts")
    return comm.alltoallv(confirm_bufs, confirm_counts, "setup:map:confirm")


def _check_confirmations(maps, halos, confs, confcounts):
    """Step 4, receive side: the requester checks the count and hash against its receive map."""
    for m, conf, cc in zip(maps, confs, confcounts):
        for k, s in enumerate(np.flatnonzero(cc)):
            n, hsh = conf[2 * k:2 * k + 2]
            ids = halos[m.rank]["global_id"][m.recv[int(s)]]
            if n != len(ids) or hsh != _id_hash(ids):
                raise AssertionError(f"rank {m.rank}: map handshake mismatch with owner {s}")


def halo_exchange(comm, maps, core_values, phase="recurring:halo"):
    """Refresh halo values from their owners using the frozen maps (no rediscovery).

    Parameters
    ----------
    core_values : per rank, a float array in owned (global_id) order
    phase : name under which the traffic is counted in comm.stats

    Returns
    -------
    per rank, a float array of length nhalo, in halo_local_index order
    """
    sends = [{d: core_values[m.rank][idx] for d, idx in m.send.items()} for m in maps]
    recvs = comm.neighbor_exchange(sends, phase)
    out = []
    for m, recv in zip(maps, recvs):
        v = np.empty(m.nhalo, dtype=float)
        for s, sl in m.recv.items():
            v[sl] = recv[s]
        out.append(v)
    return out
