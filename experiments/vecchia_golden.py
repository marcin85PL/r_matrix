"""Vecchia prototype against core + halo on the golden cases (export.CASES), per correlation group.

For each case, partition (bisection of the active data into P patches) and group: the symmetrised
core + halo operator at the class's halo width, block Vecchia with the same width (patch
sequences "id" and "colour"), and point Vecchia (orderings "maxmin" and "patch", several m).
Inactive data are left out (every operator passes them through). Also checks, per case, that
8 fixed patches on 1, 2, 4 and 7 tasks give bit-identical C~^-1 x (D2).

    python experiments/vecchia_golden.py <out.json>
"""
import json
import sys

import numpy as np

from r_matrix import vecchia as V
from r_matrix import vecchia_eval as E
from r_matrix.decomposition import BisectionDecomposition
from r_matrix.export import CASES


def gathered(b, per_patch):
    out = np.zeros(len(b.gids))
    for v, x in zip(b.vowned, per_patch):
        out[np.searchsorted(b.gids, v["global_id"])] = x
    return out


def d2_check(obs, dec, variant, params, h):
    rng = np.random.default_rng(5)
    ref = None
    for t in (1, 2, 4, 7):
        b = E.build(obs, dec, variant, params, n_tasks=t, m=10, h=h, seq="colour")
        if ref is None:
            x = [rng.standard_normal(len(v)) for v in b.vowned]
        y = gathered(b, V.apply_inv(b.comm, b.maps, b.factors, x))
        if ref is None:
            ref = y
        elif not np.array_equal(y, ref):
            return False
    return True


def main():
    out = sys.argv[1]
    rows = []
    for name in CASES:
        obs_all, params, hs, active = E.golden_data(name)
        obs_all = obs_all[active]
        for P in (4, 8):
            dec = BisectionDecomposition.from_points(obs_all["x"], obs_all["y"], obs_all["global_id"], P)
            for g, (c, a) in params.items():
                obs = obs_all[obs_all["group"] == g]
                n = len(obs)
                if n < 2:
                    continue
                pg = {g: (c, a)}
                h = hs[g]
                C = E.exact_C(obs, c, a)
                base = dict(case=name, P=P, group=int(g), n=n, c_km=c, alpha=a, h_over_c=h / c)
                A, As, sizes, cm = E.core_halo(obs, dec, c, a, h)
                lam = V.gen_eigs_A(As, C)
                rows.append(dict(base, method="core+halo", **V.eig_metrics(lam), **E.core_halo_costs(sizes),
                                 **cm, asym_plain=float(np.abs(A - A.T).max() / np.abs(A).max())))
                for seq in ("id", "colour"):
                    b = E.build(obs, dec, "block", pg, h=h, seq=seq)
                    z = [np.ones(len(v)) for v in b.vowned]
                    rows.append(dict(base, method="block", seq=seq, **V.eig_metrics(V.gen_eigs_W(b.W(), C)),
                                     **E.costs(b), sample_rounds=V.sample(b.comm, b.maps, b.factors, z)[1]))
                for order in ("maxmin", "patch"):
                    for m in (5, 10, 20, 50):
                        if m >= n:
                            continue
                        b = E.build(obs, dec, "point", pg, m=m, order=order, R0=2 * c)
                        z = [np.ones(len(v)) for v in b.vowned]
                        rows.append(dict(base, method="point", order=order, m=m,
                                         **V.eig_metrics(V.gen_eigs_W(b.W(), C)), **E.costs(b),
                                         sample_rounds=V.sample(b.comm, b.maps, b.factors, z)[1]))
                print(name, P, g, n, flush=True)
        # D2 on 8 fixed patches, all groups at once
        dec = BisectionDecomposition.from_points(obs_all["x"], obs_all["y"], obs_all["global_id"], 8)
        hmax = max(hs.values())
        rows.append(dict(case=name, method="d2", point=d2_check(obs_all, dec, "point", params, hmax),
                         block=d2_check(obs_all, dec, "block", params, hmax)))
        json.dump(rows, open(out, "w"), indent=1, default=float)
    print("done", flush=True)


if __name__ == "__main__":
    main()
