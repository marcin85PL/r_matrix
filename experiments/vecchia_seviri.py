"""Vecchia prototype against core + halo on one real SEVIRI image, with the operational partition.

The patches are the RCB partition of *all* active data of the cycle into P patches (as rcorr does),
restricted to the image (one channel, one 15-min slot): the image is one correlation group. For
each operator A ~ C^-1 the script reports

    accuracy   generalised eigenvalues of A C (max |lambda - 1|, KL divergence, smallest), the
               relative error of A d and of d^T A d (Jo) for the real normalised departures d
    implied    C~ = A^-1: max and rms error of its variances (what EDA perturbations would have)
               and the largest error of its correlations
    cost       flops of setup and of one application (largest patch and total), stored words,
               halo / core, peers, bytes and messages of setup and of one application
    sampling   rounds of the forward substitution x = W^-1 z (Vecchia only)

    python experiments/vecchia_seviri.py <data.npz> <channel> <slot> <alpha> <P> <out.json> [quick]

data.npz from the extraction script (lon, lat in radians, seqno, channel, slot, d = fg_depar /
sigma_o). c = 115 km.
"""
import json
import sys
import time

import numpy as np
from scipy.linalg import solve_triangular

from r_matrix import vecchia as V
from r_matrix import vecchia_eval as E
from r_matrix.decomposition import BisectionDecomposition
from r_matrix.records import OBS_DTYPE

C_KM = 115.0


def load(path, channel, slot):
    z = np.load(path)
    gid_all = z["seqno"] * 2 ** 16 + z["channel"]
    m = (z["channel"] == channel) & (z["slot"] == slot)
    obs = np.zeros(int(m.sum()), dtype=OBS_DTYPE)
    obs["global_id"], obs["x"], obs["y"] = gid_all[m], z["lon"][m], z["lat"][m]
    obs["value"], obs["sigma"], obs["group"] = z["d"][m], 1.0, 0
    obs = obs[np.argsort(obs["global_id"], kind="stable")]
    return obs, (z["lon"], z["lat"], gid_all)


def metrics(A_or_W, C, L, d, zref, is_W, rank=None):
    """Accuracy and implied-covariance metrics for W (C~^-1 = W^T W) or a dense symmetric A."""
    t = time.time()
    if is_W:
        W = A_or_W.toarray()
        lam = V.gen_eigs_W(W, C)
        Ad = W.T @ (W @ d)
        o = np.argsort(rank, kind="stable")              # W is lower triangular in rank order
        Winv = solve_triangular(W[np.ix_(o, o)], np.eye(len(o)), lower=True)
        Ct = np.empty_like(C)
        Ct[np.ix_(o, o)] = Winv @ Winv.T
    else:
        A = A_or_W
        M = L.T @ A @ L
        lam = np.linalg.eigvalsh(0.5 * (M + M.T))
        Ad = A @ d
        Ct = np.linalg.inv(A) if lam.min() > 0 else np.full_like(C, np.nan)
    out = V.eig_metrics(lam)
    jo, jo_ref = d @ Ad, d @ zref
    out.update(jo_rel=float((jo - jo_ref) / jo_ref), z_rel=float(np.linalg.norm(Ad - zref) / np.linalg.norm(zref)),
               var_err_max=float(np.abs(np.diag(Ct) - 1).max()),
               var_err_rms=float(np.sqrt(np.mean((np.diag(Ct) - 1) ** 2))),
               cov_err_max=float(np.abs(Ct - C).max()), sec_metrics=time.time() - t)
    return out


def apply_comm(b):
    """Bytes, messages and busiest-patch peers of one C~^-1 x on the built operator."""
    before = {k: (v.bytes, v.messages) for k, v in b.comm.stats.phases.items()}
    x = [np.ones(len(v)) for v in b.vowned]
    V.apply_inv(b.comm, b.maps, b.factors, x, tag="apply")
    st = b.comm.stats
    by = st.total("apply")["bytes"] - sum(v[0] for k, v in before.items() if k.startswith("apply"))
    ms = st.total("apply")["messages"] - sum(v[1] for k, v in before.items() if k.startswith("apply"))
    return dict(bytes_apply=int(by), messages_apply=int(ms))


def main():
    path, channel, slot, alpha, P, out = sys.argv[1:7]
    quick = len(sys.argv) > 7
    channel, slot, alpha, P = int(channel), int(slot), float(alpha), int(P)
    obs, (lon, lat, gid_all) = load(path, channel, slot)
    n = len(obs)
    print(f"channel {channel} slot {slot}: n = {n}, alpha {alpha}, P {P}", flush=True)
    dec = BisectionDecomposition.from_points(lon, lat, gid_all, P)
    params = {0: (C_KM, alpha)}
    C = E.exact_C(obs, C_KM, alpha)
    L = np.linalg.cholesky(C)
    d = obs["value"]
    zref = np.linalg.solve(C, d)
    res = dict(channel=channel, slot=slot, alpha=alpha, P=P, n=n, c_km=C_KM, data=str(path), rows=[])

    def save(row):
        res["rows"].append(row)
        print(json.dumps(row), flush=True)
        json.dump(res, open(out, "w"), indent=1)

    # core + halo (operational: h = 4 c), symmetrised and plain
    for hc in ((4.0,) if quick else (2.0, 3.0, 4.0)):
        t = time.time()
        A, As, sizes, cm = E.core_halo(obs, dec, C_KM, alpha, hc * C_KM)
        ts = time.time() - t
        row = dict(method="core+halo", h_over_c=hc, sec_setup=ts, **E.core_halo_costs(sizes), **cm,
                   asym_plain=float(np.abs(A - A.T).max() / np.abs(A).max()),
                   ncore_max=int(max(s[0] for s in sizes)))
        row.update(metrics(As, C, L, d, zref, False))
        save(row)

    # block Vecchia: patches as blocks, one-sided halo of width h
    for seq in ("colour", "id"):
        for hc in ((4.0,) if quick else (2.0, 3.0, 4.0, 6.0)):
            if seq == "id" and hc != 4.0:
                continue
            t = time.time()
            b = E.build(obs, dec, "block", params, seq=seq, h=hc * C_KM)
            ts = time.time() - t
            row = dict(method="block", seq=seq, h_over_c=hc, sec_setup=ts, **E.costs(b), **apply_comm(b))
            z = [np.random.default_rng(0).standard_normal(len(v)) for v in b.vowned]
            row["sample_rounds"] = V.sample(b.comm, b.maps, b.factors, z)[1]
            rank = np.concatenate([v["rank"] for v in b.vowned])
            rgid = np.concatenate([v["global_id"] for v in b.vowned])
            rank = rank[np.argsort(rgid)]
            row.update(metrics(b.W(), C, L, d, zref, True, rank))
            save(row)

    # point Vecchia
    for order in ("maxmin", "patch"):
        for m in ((50,) if quick else (30, 50, 100)):
            t = time.time()
            b = E.build(obs, dec, "point", params, m=m, order=order, R0=4 * C_KM)
            ts = time.time() - t
            row = dict(method="point", order=order, m=m, sec_setup=ts, **E.costs(b), **apply_comm(b),
                       setup_per_round=b.info["per_round"])
            z = [np.random.default_rng(0).standard_normal(len(v)) for v in b.vowned]
            row["sample_rounds"] = V.sample(b.comm, b.maps, b.factors, z)[1]
            rank = np.concatenate([v["rank"] for v in b.vowned])
            rgid = np.concatenate([v["global_id"] for v in b.vowned])
            rank = rank[np.argsort(rgid)]
            row.update(metrics(b.W(), C, L, d, zref, True, rank))
            save(row)


if __name__ == "__main__":
    main()
