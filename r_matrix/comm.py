"""Serial mock of the MPI collectives used by the prototype.

Nothing here runs in parallel. A "rank" is simply an index into a Python list, and
every method takes the inputs of *all* ranks at once, as a list indexed by rank, and
returns the outputs of all ranks. The code therefore reads like SPMD code with an
outer ``for rank in range(P)`` loop made explicit.

The semantics (buffers packed by destination, explicit counts, receive buffers
packed by source) match MPI one to one:

=========================  ==================================================
MockComm method            MPI equivalent
=========================  ==================================================
alltoall(sendcounts)       MPI_Alltoall of one int64 per peer
alltoallv(bufs, counts)    MPI_Alltoallv with explicit counts/displacements
neighbor_exchange(sends)   persistent Isend/Irecv or MPI_Neighbor_alltoallv
=========================  ==================================================

Porting to real MPI (e.g. mpi4py) keeps the algorithm and the data layouts, but it
is not a drop-in swap of this class. In MPI each process holds only its own buffers,
so the ``for r in range(P)`` loops around the calls in the other modules become the
body that process ``r`` runs, and each ``MockComm`` call becomes one collective
called by every process.

Every call is tagged with a *phase* name, such as "setup:halo:records" or
"recurring:halo", and ``CommStats`` accumulates per phase:

* bytes: the off-rank payload, summed over all (source, destination) pairs;
* messages: the number of non-empty off-rank (source, destination) pairs;
* max_peers: the largest number of distinct non-empty destinations on any one rank.

Self-sends (rank to itself) are free, as they would be in a real code. Phases whose
names start with "setup" happen once per observation geometry. Phases whose names
start with "recurring" happen at every application of R^-1, and they are the cost
that matters inside a minimisation.
"""
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhaseStats:
    """Accumulated communication counters for one named phase."""
    calls: int = 0
    bytes: int = 0
    messages: int = 0
    max_peers: int = 0


@dataclass
class CommStats:
    """Communication counters of every phase seen by a MockComm."""
    phases: dict = field(default_factory=lambda: defaultdict(PhaseStats))

    def record(self, phase, pair_bytes):
        """Add one call; pair_bytes[s][d] is the number of bytes rank s sends to rank d."""
        st = self.phases[phase]
        st.calls += 1
        pb = np.asarray(pair_bytes, dtype=np.int64).copy()
        np.fill_diagonal(pb, 0)                     # self-sends are not communication
        st.bytes += int(pb.sum())
        st.messages += int((pb > 0).sum())
        st.max_peers = max(st.max_peers, int((pb > 0).sum(axis=1).max(initial=0)))

    def table(self):
        """{phase: {calls, bytes, messages, max_peers}}; convenient for a DataFrame."""
        return {k: vars(v).copy() for k, v in self.phases.items()}

    def total(self, prefix=""):
        """Sum of bytes and messages over all phases whose name starts with prefix."""
        sel = [v for k, v in self.phases.items() if k.startswith(prefix)]
        return {"bytes": sum(v.bytes for v in sel), "messages": sum(v.messages for v in sel)}


class MockComm:
    """A P-rank communicator simulated in one process."""

    def __init__(self, size):
        self.size = size
        self.stats = CommStats()

    def alltoall(self, sendcounts, phase):
        """Exchange one integer per rank pair (typically message sizes).

        Parameters
        ----------
        sendcounts : list of P int arrays of length P
            sendcounts[s][d] is what rank s tells rank d.

        Returns
        -------
        list of P int arrays; element [d][s] equals sendcounts[s][d].
        """
        sc = np.asarray(sendcounts, dtype=np.int64).reshape(self.size, self.size)
        self.stats.record(phase, np.full_like(sc, 8))
        return [sc[:, d].copy() for d in range(self.size)]

    def alltoallv(self, sendbufs, sendcounts, phase):
        """Variable-size all-to-all of typed arrays.

        Parameters
        ----------
        sendbufs : list of P numpy arrays (any dtype, e.g. OBS_DTYPE or float64)
            sendbufs[s] must be *packed by destination*: first the items for rank 0,
            then those for rank 1, and so on. The displacements are the cumulative
            sums of the counts, as with MPI_Alltoallv.
        sendcounts : list of P int arrays of length P
            sendcounts[s][d] is the number of items rank s sends to rank d.

        Returns
        -------
        recvbufs : list of P arrays; recvbufs[d] is packed by source rank.
        recvcounts : list of P int arrays; recvcounts[d][s] is the number of items
            rank d received from rank s.
        """
        P = self.size
        sc = np.asarray(sendcounts, dtype=np.int64).reshape(P, P)
        sdispl = np.zeros((P, P + 1), dtype=np.int64)
        sdispl[:, 1:] = np.cumsum(sc, axis=1)
        for s in range(P):
            if sdispl[s, -1] != len(sendbufs[s]):
                raise ValueError(f"rank {s}: sendcounts do not match buffer length")
        itemsize = sendbufs[0].dtype.itemsize
        self.stats.record(phase, sc * itemsize)
        recvbufs = []
        for d in range(P):
            parts = [sendbufs[s][sdispl[s, d]:sdispl[s, d + 1]] for s in range(P)]
            recvbufs.append(np.concatenate(parts))
        return recvbufs, [sc[:, d].copy() for d in range(P)]

    def neighbor_exchange(self, sends, phase):
        """Sparse point-to-point exchange, used for the recurring halo update.

        Parameters
        ----------
        sends : list of P dicts {destination_rank: array}
            Only real neighbours appear, and empty arrays are skipped, so zero-count
            peers cost nothing (unlike an all-to-all).

        Returns
        -------
        list of P dicts {source_rank: array}.
        """
        P = self.size
        pair_bytes = np.zeros((P, P), dtype=np.int64)
        recvs = [{} for _ in range(P)]
        for s in range(P):
            for d, buf in sends[s].items():
                if len(buf) == 0:
                    continue
                pair_bytes[s, d] = buf.nbytes
                recvs[d][s] = buf.copy()
        self.stats.record(phase, pair_bytes)
        return recvs
