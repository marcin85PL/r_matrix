"""Cost and communication of core + halo, block Vecchia and point Vecchia against the number of
patches (one patch per task), on all active data of one 12-h SEVIRI cycle (36 groups =
channel x 15-min slot), with the operational partition (RCB of all positions).

Structural counts only (no factorisation): per patch, summed over its groups,

    core + halo    halo = data of the group within h of a core datum, owned elsewhere (both sides)
                   setup sum n^3 / 3, apply (symmetrised) sum 4 n^2, n = core + halo per group
                   (two right-hand sides through the Cholesky factor, as rcorr does); "explicit":
                   the core rows of R_loc^-1 stored, apply 4 n_core n, setup + 2 n^2 n_core
    block          one-sided halo: only data of patches earlier in the sequence (greedy colouring
                   of the patch graph); setup n_N^3/3 + 2 n_N^2 n_B + 3 n_B^2 n_N + 2 n_B^3 / 3;
                   apply 4 nnz(W) with nnz = n_B (n_B + 1) / 2 + n_B n_N; sampling rounds = longest
                   chain of patch dependencies. Also with the patches in id order (h = 4 c).
    point          m nearest earlier neighbours in the max-min ordering of each group; remote
                   neighbours = the apply halo; setup m^3 / 3 + 2 m^2 per datum; apply 4 (m + 1)
                   per datum; setup rounds of the radius-doubling search from R0 = 4 c; sampling
                   rounds = longest chain of cross-patch dependencies

Bytes per application: 8 per halo value, twice (halo exchange + adjoint halo exchange).

    python experiments/vecchia_scaling.py <data.npz> <out.json> [P ...]
"""
import json
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from r_matrix import vecchia as V
from r_matrix.decomposition import BisectionDecomposition
from r_matrix.geometry import points

C_KM = 115.0


def pairs_within(xyz, grp, h):
    """All ordered pairs (i, j), i != j, of the same group within chord h."""
    out = []
    for g in np.unique(grp):
        idx = np.flatnonzero(grp == g)
        pr = cKDTree(xyz[idx]).query_pairs(h, output_type="ndarray")
        out.append(idx[pr])
    pr = np.concatenate(out)
    return np.concatenate([pr, pr[:, ::-1]])


def colour_sequence(own, pairs, P):
    """Greedy colouring of the patch graph (patches holding data within h of each other)."""
    a, b = own[pairs[:, 0]], own[pairs[:, 1]]
    k = a != b
    e = np.unique(a[k] * P + b[k])
    adj = [[] for _ in range(P)]
    for x in e:
        adj[x // P].append(x % P)
    colour = np.full(P, -1)
    for p in range(P):
        used = {colour[q] for q in adj[p] if colour[q] >= 0}
        colour[p] = next(c for c in range(P + 1) if c not in used)
    return np.lexsort((np.arange(P), colour)), colour, np.array([len(x) for x in adj])


def summary(x):
    x = np.asarray(x, dtype=float)
    return dict(max=float(x.max()), mean=float(x.mean()), p90=float(np.percentile(x, 90)))


def main():
    path, out = sys.argv[1], sys.argv[2]
    Ps = [int(p) for p in sys.argv[3:]] or [96, 192, 384, 768, 1024]
    z = np.load(path)
    lon, lat = z["lon"], z["lat"]
    gid = z["seqno"] * 2 ** 16 + z["channel"]
    gkey = z["channel"] * 100 + z["slot"]
    groups, grp = np.unique(gkey, return_inverse=True)
    xyz = points(lon, lat, "sphere")
    n, G = len(gid), len(groups)
    print(f"n = {n}, groups = {G}", flush=True)
    res = dict(source=str(path), n=int(n), n_groups=int(G), c_km=C_KM, results=[])

    # ---- point Vecchia neighbour sets (independent of P) --------------------------------------
    t = time.time()
    point = {}
    rank = np.zeros(n, dtype=np.int64)
    for g in range(G):
        idx = np.flatnonzero(grp == g)
        o = V.maxmin_order(xyz[idx], gid[idx])[0]
        rank[idx] = V.rank_of(o)
    for m in (50, 100):
        nb = [None] * n
        reach = np.zeros(n)
        for g in range(G):
            idx = np.flatnonzero(grp == g)
            loc, rch = V.point_neighbours(xyz[idx], rank[idx], m)
            for k, i in enumerate(idx):
                nb[i] = idx[loc[k]]
            reach[idx] = rch
        point[m] = (nb, reach)
        print(f"point m={m} neighbours {time.time() - t:.0f}s", flush=True)
    pairs = {hc: pairs_within(xyz, grp, hc * C_KM) for hc in (3.0, 4.0)}
    print(f"pairs {time.time() - t:.0f}s", flush=True)

    for P in Ps:
        t = time.time()
        dec = BisectionDecomposition.from_points(lon, lat, gid, P)
        own = dec.owner(lon, lat)
        core_pg = np.bincount(own * G + grp, minlength=P * G).astype(float)
        row = dict(P=P, patch_diam_km=float(2 * 6371 * np.median(dec.radii)), core_per_patch=n / P)

        # core + halo (h = 4 c), both sides
        pr = pairs[4.0]
        oi, oj = own[pr[:, 0]], own[pr[:, 1]]
        k = oi != oj
        key = np.unique(oi[k].astype(np.int64) * n + pr[k, 1])
        hp, hj = key // n, key % n
        halo_pg = np.bincount(hp * G + grp[hj], minlength=P * G).astype(float)
        nn = core_pg + halo_pg
        peers = np.array([len(np.unique(own[hj[hp == p]])) for p in range(P)])
        ch = dict(halo_over_core=float(halo_pg.sum() / n),
                  flops_setup=summary((nn ** 3 / 3).reshape(P, G).sum(1)),
                  flops_apply=summary((4 * nn ** 2).reshape(P, G).sum(1)),
                  words=summary((nn ** 2 / 2).reshape(P, G).sum(1)),
                  bytes_apply=summary(16 * halo_pg.reshape(P, G).sum(1)), peers=summary(peers))
        row["core_halo"] = ch
        # the same operator with the core rows of R_loc^-1 stored explicitly (n_core x n): setup
        # adds the n_core unit solves (2 n^2 n_core); apply = A x and A^T x, 2 n_core n flops each
        nc = core_pg
        row["core_halo_explicit"] = dict(ch, flops_setup=summary((nn ** 3 / 3 + 2 * nn ** 2 * nc).reshape(P, G).sum(1)),
                                         flops_apply=summary((4 * nc * nn).reshape(P, G).sum(1)),
                                         words=summary((nc * nn).reshape(P, G).sum(1)))

        # block Vecchia, one-sided halo, colour sequence (patch graph from h = 4 c)
        seq, colour, deg = colour_sequence(own, pairs[4.0], P)
        row["colours"] = int(colour.max() + 1)
        row["patch_degree"] = summary(deg)
        for hc, how in ((3.0, "colour"), (4.0, "colour"), (4.0, "id")):
            pr = pairs[hc]
            oi, oj = own[pr[:, 0]], own[pr[:, 1]]
            pos = np.empty(P, dtype=np.int64)
            pos[seq if how == "colour" else np.arange(P)] = np.arange(P)
            k = pos[oj] < pos[oi]
            key = np.unique(oi[k].astype(np.int64) * n + pr[k, 1])
            hp, hj = key // n, key % n
            nN = np.bincount(hp * G + grp[hj], minlength=P * G).astype(float)
            nB = core_pg
            fs = nN ** 3 / 3 + 2 * nN ** 2 * nB + 3 * nB ** 2 * nN + 2 * nB ** 3 / 3
            nnz = nB * (nB + 1) / 2 + nB * nN
            dep = [np.unique(own[hj[hp == p]]) for p in range(P)]
            rounds = np.zeros(P, dtype=np.int64)
            for p in (seq if how == "colour" else range(P)):
                if len(dep[p]):
                    rounds[p] = rounds[dep[p]].max() + 1
            row[f"block_h{hc:g}" + ("_id" if how == "id" else "")] = dict(
                halo_over_core=float(nN.sum() / n), flops_setup=summary(fs.reshape(P, G).sum(1)),
                flops_apply=summary((4 * nnz).reshape(P, G).sum(1)), words=summary(nnz.reshape(P, G).sum(1)),
                bytes_apply=summary(16 * nN.reshape(P, G).sum(1)),
                peers=summary([len(x) for x in dep]), sample_rounds=int(rounds.max()))

        # point Vecchia
        for m, (nb, reach) in point.items():
            src = np.repeat(np.arange(n), [len(x) for x in nb])
            dst = np.concatenate(nb)
            k = own[src] != own[dst]
            key = np.unique(own[src[k]].astype(np.int64) * n + dst[k])
            hp, hj = key // n, key % n
            nh = np.bincount(hp, minlength=P).astype(float)
            peers = np.array([len(np.unique(own[hj[hp == p]])) for p in range(P)])
            kk = np.array([len(x) for x in nb], dtype=float)
            fs = np.bincount(own, weights=kk ** 3 / 3 + 2 * kk ** 2, minlength=P)
            fa = np.bincount(own, weights=4 * (kk + 1), minlength=P)
            # sampling rounds: in rank order per group, a cross-patch dependency costs one round
            rnd = np.zeros(n, dtype=np.int64)
            for g in range(G):
                idx = np.flatnonzero(grp == g)
                for i in idx[np.argsort(rank[idx])]:
                    if len(nb[i]):
                        rnd[i] = (rnd[nb[i]] + (own[nb[i]] != own[i])).max()
            R0 = 4 * C_KM
            sr = 1 + np.maximum(0, np.ceil(np.log2(np.maximum(reach, 1e-9) / R0))).astype(int)
            row[f"point_m{m}"] = dict(
                halo_over_core=float(nh.sum() / n), flops_setup=summary(fs), flops_apply=summary(fa),
                words=summary(np.bincount(own, weights=2 * (kk + 1), minlength=P)),
                bytes_apply=summary(16 * nh), peers=summary(peers), sample_rounds=int(rnd.max()),
                reach_km=dict(median=float(np.median(reach)), p90=float(np.percentile(reach, 90)),
                              max=float(reach.max())),
                setup_rounds=int(sr.max()),
                setup_rounds_frac={int(r): float((sr <= r).mean()) for r in range(1, sr.max() + 1)})
        row["sec"] = time.time() - t
        res["results"].append(row)
        print(json.dumps(row), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
