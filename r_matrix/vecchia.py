"""Sparse inverse-Cholesky (Vecchia) approximations of C^-1, serial and on mock MPI tasks.

The core + halo operator (covariance.LocalSolver) approximates C^-1 d patch by patch and is
symmetrised afterwards. This module prototypes the alternative of design note 3, section 2.2:
an approximate *factor* W of the precision,

    C~^-1 = W^T W,      W lower triangular in some ordering of the data,

which is symmetric positive definite by construction (W has a positive diagonal, so it is
invertible) and gives, from one factor, R^-1 = W^T W (4D-Var), R^-1/2 = W (eigenProjection,
dual minimisers) and R^1/2 z = W^-1 z (EDA perturbations, by forward substitution). Here
R = C (sigma = 1); with sigma, R^-1 = S^-1 W^T W S^-1.

Two variants, both computed per correlation group (data of different groups are uncorrelated):

point   Vecchia (1988), "factorised sparse approximate inverse" (FSAI) in linear algebra. The
        data are ordered (``maxmin_order``: coarse to fine; or ``patch_maxmin_order``) and every
        datum i is regressed on its m nearest *earlier* neighbours N(i):

            b_i = C_NN^-1 C_N,i,   d_i = 1 - C_i,N b_i,   (W x)_i = (x_i - b_i . x_N(i)) / sqrt(d_i)

        so row i of W has m + 1 nonzeros. Exact (W^T W = C^-1) when N(i) is every earlier datum.

block   Block Vecchia (Katzfuss and Guinness 2021, section 3) or block FSAI (Janna et al. 2010)
        with the R-patches as blocks: the patches are put in a sequence, and the data B of a
        patch are conditioned *jointly* on N(B), the data of earlier patches within distance h
        of B (a one-sided halo):

            S = C_BN C_NN^-1,   C_BB - S C_NB = L L^T,   (W x)_B = L^-1 (x_B - S x_N)

        Exact when N(B) is all earlier data. Its cost and communication are those of the core +
        halo operator with half the halo.

Every builder returns, per patch, a ``LocalFactor``: the owned rows of W as a CSR matrix over
the patch's local vector [own data (sorted by global ID); halo (sorted by owner, global ID)].
The recurring operations are then generic:

    apply_W        z = W x               one halo exchange              (R^-1/2)
    apply_Wt       y = W^T z             one adjoint halo exchange
    apply_inv      y = W^T W x           both                           (R^-1)
    sample         x = W^-1 z            forward substitution in rounds (R^1/2)

Determinism (D2): the ordering is a function of the data (and, for patch-based orderings, of
the fixed patch layout); every reduction is in (patch, global ID) order, so results do not
depend on the number of tasks.
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.spatial import cKDTree

from .commmap import build_comm_maps, halo_exchange, halo_exchange_adjoint
from .covariance import gaspari_cohn
from .geometry import EARTH_RADIUS, points

# Setup record: what a patch needs to know about a datum (its own or another patch's).
VREC = np.dtype([("global_id", "i8"), ("x", "f8"), ("y", "f8"), ("group", "i4"), ("rank", "i8")])


# ---------------------------------------------------------------------------------------------
# Correlations and orderings
# ---------------------------------------------------------------------------------------------

def dist(a, b):
    """Chord distances between the rows of a (k, 3) and b (l, 3); the one formula used for every
    neighbour selection, so that serial and distributed selections agree bit for bit."""
    return np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))


def corr(a, b, c, alpha, same=False):
    """alpha GC(chord; c), plus the nugget (1 - alpha) I when a and b are the same point set."""
    out = alpha * gaspari_cohn(dist(a, b), c)
    if same:
        out[np.diag_indices_from(out)] += 1.0 - alpha
    return out


def maxmin_order(xyz, gid):
    """Greedy max-min (farthest point) ordering, coarse to fine (Guinness 2018).

    Starts at the datum nearest the centroid, then repeatedly takes the datum farthest from all
    those already taken. Ties go to the smallest global ID, so the ordering is a function of the
    data only. O(n^2) time, O(n) memory.

    Returns (order, ell): order[k] is the index (into xyz) of the k-th datum, and ell[k] its
    distance to the earlier data when it was taken (inf for the first).
    """
    n = len(gid)
    if n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    srt = np.argsort(gid, kind="stable")
    p = xyz[srt]
    k = int(np.argmin(np.linalg.norm(p - p.mean(0), axis=1)))
    order, ell = [k], [np.inf]
    dmin = np.linalg.norm(p - p[k], axis=1)
    taken = np.zeros(n, dtype=bool)
    taken[k] = True
    for _ in range(n - 1):
        dm = np.where(taken, -1.0, dmin)
        k = int(np.argmax(dm))
        order.append(k)
        ell.append(dm[k])
        taken[k] = True
        dmin = np.minimum(dmin, np.linalg.norm(p - p[k], axis=1))
    return srt[np.array(order)], np.array(ell)


def patch_maxmin_order(xyz, gid, patch, seq):
    """Patches in the sequence seq, max-min inside each patch (a D2-friendly ordering that a
    patch can compute from its own data)."""
    out = []
    for p in seq:
        idx = np.flatnonzero(patch == p)
        out.append(idx[maxmin_order(xyz[idx], gid[idx])[0]])
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def rank_of(order):
    """Inverse permutation: the position of every datum in the ordering."""
    r = np.empty(len(order), dtype=np.int64)
    r[order] = np.arange(len(order))
    return r


def patch_sequence(xyz, patch, n_patches, h, how="id"):
    """The order of the patches for the block variant (and for patch_maxmin_order).

    id      ascending patch id
    colour  greedy colouring of the patch graph (an edge where two patches hold data within h of
            each other) in patch-id order, then by (colour, id). Patches of one colour are then
            independent in the forward substitution, so sampling needs about as many rounds as
            there are colours.
    Returns (seq, colour); colour is None for "id".
    """
    if how == "id":
        return np.arange(n_patches), None
    if how != "colour":
        raise ValueError(f"unknown patch sequence {how!r}")
    adj = patch_adjacency(xyz, patch, n_patches, h)
    colour = np.full(n_patches, -1)
    for p in range(n_patches):
        used = {colour[q] for q in adj[p] if colour[q] >= 0}
        colour[p] = next(k for k in range(n_patches + 1) if k not in used)
    return np.lexsort((np.arange(n_patches), colour)), colour


def patch_adjacency(xyz, patch, n_patches, h):
    """For each patch, the set of other patches holding data within distance h of its data."""
    pairs = cKDTree(xyz).query_pairs(h * (1 + 1e-12), output_type="ndarray")
    adj = [set() for _ in range(n_patches)]
    if len(pairs):
        a, b = patch[pairs[:, 0]], patch[pairs[:, 1]]
        for p, q in set(zip(a[a != b].tolist(), b[a != b].tolist())):
            adj[p].add(q)
            adj[q].add(p)
    return adj


# ---------------------------------------------------------------------------------------------
# Serial factors (the references for the distributed builders)
# ---------------------------------------------------------------------------------------------

def select_earlier(x_i, rank_i, cand_xyz, cand_rank, m, cap=np.inf):
    """The (at most) m nearest candidates with rank < rank_i and distance <= cap.

    Ties in distance go to the smaller rank. Returns (indices into the candidates, distances),
    nearest first.
    """
    d = dist(x_i[None, :], cand_xyz)[0]
    ok = np.flatnonzero((cand_rank < rank_i) & (d <= cap))
    if len(ok) > m:
        thr = np.partition(d[ok], m - 1)[m - 1]
        ok = ok[d[ok] <= thr]
    ok = ok[np.lexsort((cand_rank[ok], d[ok]))][:m]
    return ok, d[ok]


def point_neighbours(xyz, rank, m, cap=np.inf):
    """N(i) for every datum (indices into xyz, nearest first) and the distance to the farthest."""
    nb, reach = [], np.zeros(len(rank))
    for i in range(len(rank)):
        idx, d = select_earlier(xyz[i], rank[i], xyz, rank, m, cap)
        nb.append(idx)
        reach[i] = d[-1] if len(d) else 0.0
    return nb, reach


def point_coeffs(x_i, x_nb, c, alpha):
    """Regression weights b and conditional variance d of one datum on its neighbours."""
    if len(x_nb) == 0:
        return np.zeros(0), 1.0
    cnn = corr(x_nb, x_nb, c, alpha, same=True)
    cni = corr(x_nb, x_i[None, :], c, alpha)[:, 0]
    b = cho_solve(cho_factor(cnn, lower=True), cni)
    return b, 1.0 - cni @ b


def point_W(xyz, nb, c, alpha):
    """Sparse W (n x n, rows and columns in the input order) of the point variant."""
    rows, cols, vals = [], [], []
    for i, n_i in enumerate(nb):
        b, d = point_coeffs(xyz[i], xyz[n_i], c, alpha)
        s = d ** -0.5
        rows += [i] * (len(n_i) + 1)
        cols += [i, *n_i.tolist()]
        vals += [s, *(-b * s).tolist()]
    n = len(nb)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


def block_sets(xyz, gid, patch, seq, h):
    """Conditioning set of every patch for the block variant: data of earlier patches (in seq)
    within distance h of any of its data, sorted by global ID. Returns {patch: indices}."""
    pos = np.empty(max(seq) + 1 if len(seq) else 0, dtype=np.int64)
    pos[seq] = np.arange(len(seq))
    tree = cKDTree(xyz)
    out = {}
    for p in seq:
        own = np.flatnonzero(patch == p)
        if len(own) == 0:
            out[p] = own
            continue
        hits = np.unique(np.concatenate(tree.query_ball_point(xyz[own], r=h * (1 + 1e-9)))
                         .astype(np.int64))
        hits = hits[pos[patch[hits]] < pos[p]]
        near = np.zeros(len(hits), dtype=bool)
        for j0 in range(0, len(hits), 512):
            near[j0:j0 + 512] = (dist(xyz[hits[j0:j0 + 512]], xyz[own]) <= h).any(1)
        hits = hits[near]
        out[p] = hits[np.argsort(gid[hits], kind="stable")]
    return out


def block_coeffs(x_b, x_n, c, alpha):
    """W blocks of one patch: (W_BB, W_BN) = (L^-1, -L^-1 S), with S = C_BN C_NN^-1 and
    L L^T = C_BB - S C_NB. Rows and columns in the order given."""
    cbb = corr(x_b, x_b, c, alpha, same=True)
    if len(x_n) == 0:
        return solve_triangular(cholesky(cbb, lower=True), np.eye(len(x_b)), lower=True), \
            np.zeros((len(x_b), 0))
    cnn = corr(x_n, x_n, c, alpha, same=True)
    cbn = corr(x_b, x_n, c, alpha)
    S = cho_solve(cho_factor(cnn, lower=True), cbn.T).T
    L = cholesky(cbb - S @ cbn.T, lower=True)
    wbb = solve_triangular(L, np.eye(len(x_b)), lower=True)
    return wbb, -wbb @ S


def block_W(xyz, gid, patch, seq, sets, c, alpha):
    """Sparse W (n x n, input order) of the block variant; data of a patch ordered by global ID."""
    n = len(gid)
    W = sp.lil_matrix((n, n))
    for p in seq:
        own = np.flatnonzero(patch == p)
        if len(own) == 0:
            continue
        own = own[np.argsort(gid[own], kind="stable")]
        wbb, wbn = block_coeffs(xyz[own], xyz[sets[p]], c, alpha)
        W[np.ix_(own, own)] = wbb
        if len(sets[p]):
            W[np.ix_(own, sets[p])] = wbn
    return W.tocsr()


def block_rank(gid, patch, seq):
    """An ordering in which the block W is lower triangular: by (position of patch, global ID)."""
    pos = np.empty(max(seq) + 1, dtype=np.int64)
    pos[seq] = np.arange(len(seq))
    return rank_of(np.lexsort((gid, pos[patch])))


# ---------------------------------------------------------------------------------------------
# Distributed: per-patch factors on a MockComm
# ---------------------------------------------------------------------------------------------

@dataclass
class LocalFactor:
    """The owned rows of W on one patch.

    W_loc : CSR, ncore x (ncore + nhalo), over the local vector [own; halo]
    rank  : ordering key of the own data (W is lower triangular in rank), for sampling
    halo_rank : ordering key of the halo data
    flops_setup : floating-point operations of the setup (Cholesky m^3/3 or n^3/3, triangular
        solves, products; correlation evaluations not counted)
    """
    W_loc: sp.csr_matrix
    rank: np.ndarray
    halo_rank: np.ndarray
    flops_setup: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def ncore(self):
        return self.W_loc.shape[0]

    @property
    def nhalo(self):
        return self.W_loc.shape[1] - self.W_loc.shape[0]

    @property
    def flops_apply(self):
        """Flops of one product with W (or W^T): two per stored nonzero."""
        return 2.0 * self.W_loc.nnz


def to_vrec(owned, rank_by_gid):
    """VREC copies of owned OBS_DTYPE records, with their ordering key from a gid -> rank map."""
    out = []
    for o in owned:
        v = np.zeros(len(o), dtype=VREC)
        for f in ("global_id", "x", "y", "group"):
            v[f] = o[f]
        v["rank"] = [rank_by_gid[g] for g in o["global_id"]] if len(o) else []
        out.append(v)
    return out


def push_within(comm, decomp, owned, targets, R, phase, allow=None):
    """Every patch receives the records of other patches within distance R of its target data.

    A geometric push from the owners (as halo.discover_halos, criterion "obs"), restricted to
    receivers with targets and, if allow is given, to pairs with allow(src, dst) true. Returns,
    per patch, the received records (deduplicated, sorted by (owner, global ID)) and their owners.
    """
    P = comm.size
    active = np.array([len(t) > 0 for t in targets])
    sendbufs, sendcounts = [], []
    for s, o in enumerate(owned):
        parts, counts = [], np.zeros(P, dtype=np.int64)
        for d in decomp.candidate_ranks(s, R) if len(o) else []:
            if d == s or not active[d] or (allow is not None and not allow(s, d)):
                continue
            msk = decomp.push_mask(d, o["x"], o["y"], R)
            parts.append((d, o[msk]))
            counts[d] = msk.sum()
        parts.sort(key=lambda t: t[0])
        sendbufs.append(np.concatenate([q for _, q in parts]) if parts else owned[0][:0])
        sendcounts.append(counts)
    comm.alltoall(sendcounts, phase + ":counts")
    recv, rcounts = comm.alltoallv(sendbufs, sendcounts, phase + ":records")
    out, own_of = [], []
    for r in range(P):
        src = np.repeat(np.arange(P), rcounts[r])
        c = recv[r]
        keep = np.zeros(len(c), dtype=bool)
        if len(c) and len(targets[r]):
            tree = cKDTree(points(c["x"], c["y"], "sphere"))
            for hits in tree.query_ball_point(points(targets[r]["x"], targets[r]["y"], "sphere"),
                                              r=R * (1 + 1e-9)):
                keep[hits] = True
        c, src = c[keep], src[keep]
        _, first = np.unique(c["global_id"], return_index=True)
        c, src = c[first], src[first]
        o = np.lexsort((c["global_id"], src))
        out.append(c[o])
        own_of.append(src[o])
    return out, own_of


def allgather_records(comm, parts, phase):
    """Every patch receives every other patch's records (MPI_Allgatherv), sorted by global ID."""
    P = comm.size
    sendbufs = [np.concatenate([p] * (P - 1)) if P > 1 else p[:0] for p in parts]
    counts = [np.array([0 if d == s else len(parts[s]) for d in range(P)], dtype=np.int64)
              for s in range(P)]
    # alltoallv needs the buffer packed by destination: identical copies, one per other rank
    comm.alltoall(counts, phase + ":counts")
    recv, _ = comm.alltoallv(sendbufs, counts, phase + ":records")
    return [np.concatenate([recv[r], parts[r]])[np.argsort(np.concatenate([recv[r], parts[r]])
                                                         ["global_id"], kind="stable")]
            for r in range(P)]


def setup_point(comm, decomp, vowned, m, params, R0, coarse=0, cap=np.inf, max_rounds=12):
    """Distributed setup of the point variant.

    vowned : per patch, VREC records (sorted by global ID) with their global ordering key
    params : {group: (c, alpha)}
    R0     : radius of the first position halo; doubled in each further round, for the data whose
             m nearest earlier neighbours are not yet certified
    coarse : the first `coarse` data of every group's ordering are gathered on all patches
             (data with rank < coarse are certified at once: all their earlier data are known)

    Certification: once every datum within R of datum i is known, the selection on the known set
    is the exact one if its m-th distance is <= R, or if fewer than m earlier data lie within the
    cap and R >= cap, or if R spans the sphere. Returns (factors, maps, halos, halo_owner, info).
    """
    P = comm.size
    known = [v.copy() for v in vowned]
    if coarse:
        cparts = [v[v["rank"] < coarse] for v in vowned]
        cg = allgather_records(comm, cparts, "setup:vecchia:coarse")
        known = [np.unique(np.concatenate([k, c])) for k, c in zip(known, cg)]
    todo = [np.ones(len(v), dtype=bool) for v in vowned]
    nbrs = [[None] * len(v) for v in vowned]

    def certify(R):
        """Select N(i) on the known set for the uncertified data; keep the certified ones."""
        ncert = 0
        for p, v in enumerate(vowned):
            kx = points(known[p]["x"], known[p]["y"], "sphere")
            for i in np.flatnonzero(todo[p]):
                ci = np.flatnonzero(known[p]["group"] == v["group"][i])
                xi = points(v["x"][i:i + 1], v["y"][i:i + 1], "sphere")[0]
                idx, d = select_earlier(xi, v["rank"][i], kx[ci], known[p]["rank"][ci], m, cap)
                full = len(idx) == m and d[-1] <= R * (1 - 1e-9)
                short = len(idx) < m and R >= cap
                if full or short or R >= 2.0 * EARTH_RADIUS or v["rank"][i] < coarse:
                    nbrs[p][i] = known[p][ci[idx]]
                    todo[p][i] = False
                    ncert += 1
        return ncert

    per_round = [dict(R=0.0, certified=certify(0.0))] if coarse else []
    R, rounds = R0, 0
    while any(t.any() for t in todo):
        if rounds == max_rounds:
            raise RuntimeError(f"point setup: not certified after {max_rounds} rounds")
        rounds += 1
        targets = [v[t] for v, t in zip(vowned, todo)]
        recv, _ = push_within(comm, decomp, vowned, targets, R, f"setup:vecchia:r{rounds}")
        known = [np.unique(np.concatenate([k, rr])) for k, rr in zip(known, recv)]
        per_round.append(dict(R=R, certified=certify(R)))
        R *= 2.0
    # apply halo: the remote neighbours, sorted by (owner, global ID)
    halos, halo_owner = [], []
    owner_of = {}
    for p, v in enumerate(vowned):
        owner_of.update({int(g): p for g in v["global_id"]})
    for p, v in enumerate(vowned):
        rem = [nb[~np.isin(nb["global_id"], v["global_id"])] for nb in nbrs[p]] or [v[:0]]
        rem = np.unique(np.concatenate(rem)) if rem else v[:0]
        own = np.array([owner_of[int(g)] for g in rem["global_id"]], dtype=np.int64)
        o = np.lexsort((rem["global_id"], own))
        halos.append(rem[o])
        halo_owner.append(own[o])
    maps = build_comm_maps(comm, vowned, halos, halo_owner)
    factors = []
    for p, v in enumerate(vowned):
        loc = np.concatenate([v, halos[p]])
        lgid = loc["global_id"]
        srt = np.argsort(lgid, kind="stable")
        lx = points(loc["x"], loc["y"], "sphere")
        rows, cols, vals, fl = [], [], [], 0.0
        for i in range(len(v)):
            nb = nbrs[p][i]
            c, alpha = params[int(v["group"][i])]
            cidx = srt[np.searchsorted(lgid[srt], nb["global_id"])]
            b, d = point_coeffs(lx[i], lx[cidx], c, alpha)
            s = d ** -0.5
            rows += [i] * (len(cidx) + 1)
            cols += [i, *cidx.tolist()]
            vals += [s, *(-b * s).tolist()]
            k = len(cidx)
            fl += k ** 3 / 3 + 2 * k ** 2 + 2 * k
        W = sp.csr_matrix((vals, (rows, cols)), shape=(len(v), len(loc)))
        factors.append(LocalFactor(W, v["rank"].copy(), halos[p]["rank"].copy(), fl))
    info = dict(rounds=rounds, per_round=per_round)
    return factors, maps, halos, halo_owner, info


def setup_block(comm, decomp, vowned, params, h, seq):
    """Distributed setup of the block variant with the patches as blocks.

    Each patch receives, from the patches before it in seq only, their data within h of its own
    data (of the same group): a one-sided halo push. Per group it then factorises
    [halo; own] (n_N + n_B)^3 / 3 and stores W_BB = L^-1 and W_BN = -L^-1 S. Own data are in
    global-ID order, so W is lower triangular in the key (position of patch in seq, global ID),
    which is stored as the rank.
    """
    P = comm.size
    pos = np.empty(P, dtype=np.int64)
    pos[seq] = np.arange(P)
    base = 1 << 40          # rank = pos * base + index in global-ID order (a sortable key)
    vown = []
    for p, v in enumerate(vowned):
        v = v.copy()
        v["rank"] = pos[p] * base + np.arange(len(v))
        vown.append(v)
    # The halo (neighbour) records carry the owners' ranks, so exchange after setting them.
    halos, halo_owner = push_within(comm, decomp, vown, vown, h, "setup:vecchia:halo",
                                    allow=lambda s, d: pos[s] < pos[d])
    # keep only halo data of groups present on the patch (others cannot condition anything)
    for p in range(P):
        k = np.isin(halos[p]["group"], vown[p]["group"])
        # the push used all groups' own data as targets: keep same-group data within h only
        if k.any():
            hx = points(halos[p]["x"], halos[p]["y"], "sphere")
            ox = points(vown[p]["x"], vown[p]["y"], "sphere")
            for g in np.unique(halos[p]["group"][k]):
                hi = np.flatnonzero(halos[p]["group"] == g)
                oi = np.flatnonzero(vown[p]["group"] == g)
                if len(oi) == 0:
                    continue
                near = np.zeros(len(hi), dtype=bool)
                for j0 in range(0, len(hi), 512):
                    near[j0:j0 + 512] = (dist(hx[hi[j0:j0 + 512]], ox[oi]) <= h).any(1)
                k[hi[~near]] = False
        halos[p], halo_owner[p] = halos[p][k], halo_owner[p][k]
    maps = build_comm_maps(comm, vown, halos, halo_owner)
    factors = []
    for p, v in enumerate(vown):
        n_own, n_h = len(v), len(halos[p])
        W = sp.lil_matrix((n_own, n_own + n_h))
        ox = points(v["x"], v["y"], "sphere")
        hx = points(halos[p]["x"], halos[p]["y"], "sphere")
        fl, sizes = 0.0, []
        for g in np.unique(v["group"]):
            bi = np.flatnonzero(v["group"] == g)
            ni = np.flatnonzero(halos[p]["group"] == g)
            c, alpha = params[int(g)]
            wbb, wbn = block_coeffs(ox[bi], hx[ni], c, alpha)
            W[np.ix_(bi, bi)] = wbb
            if len(ni):
                W[np.ix_(bi, n_own + ni)] = wbn
            nb_, nn_ = len(bi), len(ni)
            fl += nn_ ** 3 / 3 + 2 * nn_ ** 2 * nb_ + 3 * nb_ ** 2 * nn_ + 2 * nb_ ** 3 / 3
            sizes.append((nb_, nn_))
        factors.append(LocalFactor(W.tocsr(), v["rank"].copy(), halos[p]["rank"].copy(), fl,
                                   dict(blocks=sizes)))
    return factors, maps, halos, halo_owner, vown


# ---------------------------------------------------------------------------------------------
# Recurring operations
# ---------------------------------------------------------------------------------------------

def apply_W(comm, maps, factors, x_core, phase="recurring:halo"):
    """z = W x per patch (R^-1/2 for sigma = 1): one halo exchange."""
    x_halo = halo_exchange(comm, maps, x_core, phase=phase)
    return [f.W_loc @ np.concatenate([xc, xh]) for f, xc, xh in zip(factors, x_core, x_halo)]


def apply_Wt(comm, maps, factors, z_core, phase="recurring:adj_halo"):
    """y = W^T z per patch: local transposed product, then the halo part summed on its owners."""
    loc = [f.W_loc.T @ zc for f, zc in zip(factors, z_core)]
    back = halo_exchange_adjoint(comm, maps, [l_[f.ncore:] for f, l_ in zip(factors, loc)],
                                 [f.ncore for f in factors], phase=phase)
    return [l_[:f.ncore] + b for f, l_, b in zip(factors, loc, back)]


def apply_inv(comm, maps, factors, x_core, tag="recurring"):
    """y = C~^-1 x = W^T W x: one halo exchange and one adjoint halo exchange."""
    z = apply_W(comm, maps, factors, x_core, phase=f"{tag}:halo")
    return apply_Wt(comm, maps, factors, z, phase=f"{tag}:adj_halo")


def sample(comm, maps, factors, z_core, phase="sample"):
    """x = W^-1 z (a draw from N(0, C~) when z ~ N(0, I)) by forward substitution.

    Every patch computes, in rank order, each own value whose inputs are all known; then the new
    values are sent to the patches that hold them in their halo. Repeats until done. Returns
    (x per patch, number of communication rounds). Only newly computed values travel.
    """
    x = [np.full(f.ncore, np.nan) for f in factors]
    xh = [np.full(f.nhalo, np.nan) for f in factors]
    done = [np.zeros(f.ncore, dtype=bool) for f in factors]
    sent = [np.zeros(f.ncore, dtype=bool) for f in factors]
    order = [np.argsort(f.rank, kind="stable") for f in factors]
    rounds, ndone = 0, 0
    while True:
        for p, f in enumerate(factors):
            W = f.W_loc
            for i in order[p]:
                if done[p][i]:
                    continue
                cols = W.indices[W.indptr[i]:W.indptr[i + 1]]
                vals = W.data[W.indptr[i]:W.indptr[i + 1]]
                v = np.concatenate([x[p], xh[p]])[cols]
                off = cols != i
                if np.isnan(v[off]).any():
                    continue
                x[p][i] = (z_core[p][i] - vals[off] @ v[off]) / vals[~off][0]
                done[p][i] = True
        if all(d.all() for d in done):
            break
        now = sum(int(d.sum()) for d in done)
        if now == ndone:
            raise RuntimeError("sample: no progress (W not triangular in the rank order?)")
        ndone = now
        rounds += 1
        sends = []
        for p, mp in enumerate(maps):
            new = done[p] & ~sent[p]
            sends.append({d: np.where(new[idx], x[p][idx], np.nan) for d, idx in mp.send.items()
                          if new[idx].any()})
            sent[p] |= new
        # count only the new values (as a code sending (index, value) pairs would)
        recvs = comm.neighbor_exchange([{d: b[~np.isnan(b)] for d, b in s.items()} for s in sends],
                                       f"{phase}:round")
        del recvs
        for p, mp in enumerate(maps):
            for d, buf in sends[p].items():
                ok = ~np.isnan(buf)
                sl = maps[d].recv[p]
                tgt = xh[d][sl]
                tgt[ok] = buf[ok]
                xh[d][sl] = tgt
    return x, rounds


# ---------------------------------------------------------------------------------------------
# Accuracy diagnostics against the exact C
# ---------------------------------------------------------------------------------------------

def gen_eigs_W(W, C):
    """Eigenvalues of C~^-1 C = W^T W C, i.e. of the symmetric W C W^T (1 everywhere if exact)."""
    W = W.toarray() if sp.issparse(W) else W
    M = W @ C @ W.T
    return np.linalg.eigvalsh(0.5 * (M + M.T))


def gen_eigs_A(A, C):
    """Eigenvalues of A C for a symmetric approximation A of C^-1, via L^T A L with C = L L^T."""
    L = np.linalg.cholesky(C)
    M = L.T @ A @ L
    return np.linalg.eigvalsh(0.5 * (M + M.T))


def eig_metrics(lam):
    """max |lambda - 1|, extremes, and the KL divergence between N(0, C) and N(0, C~)."""
    out = dict(lam_min=float(lam.min()), lam_max=float(lam.max()),
               max_dev=float(np.abs(lam - 1).max()), rms_dev=float(np.sqrt(np.mean((lam - 1) ** 2))),
               n_nonpos=int((lam <= 0).sum()))
    out["kl"] = float(0.5 * np.sum(lam - 1 - np.log(lam))) if lam.min() > 0 else float("nan")
    return out
