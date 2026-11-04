"""Phase A step 1: is the symmetrised core + halo operator S = (A~ + A~^T)/2 usable?

For each correlated fraction alpha and halo width h/l, it reports:
  asym          ||A~ - A~^T||_F / ||A~||_F, the asymmetry of the plain operator
  lmin_S        smallest eigenvalue of S (S is positive definite iff > 0)
  gen_min/max   range of the generalised eigenvalues of S x = lam R^-1 x. If all are
                close to 1, S is close to R^-1 in the energy norm, and the Jo Hessian
                (and so the minimisation) is barely changed.
  err_A, err_S  relative L2 error of A~ d and S d against R^-1 d (correlated d)
  bytes_A/S     communication per application (bytes, all recurring phases)

A second table repeats the main metrics for other observation patterns and
decompositions. Run (takes some minutes):

    python experiments/symmetric_operator.py > experiments/symmetric_operator.out
"""
import time

import numpy as np
from scipy.linalg import cho_solve, eigh

from r_matrix import Config, run
from r_matrix.diagnostics import approx_matrix

BASE = Config()   # 3000 obs, 4 x 3 tasks, l = c = 0.5


def analyse(cfg):
    plain = run(cfg)
    sym = run(cfg, symmetric=True)
    A = approx_matrix(plain)
    S = 0.5 * (A + A.T)
    Rinv = cho_solve(plain.ref_factor, np.eye(A.shape[0]))
    Rinv = 0.5 * (Rinv + Rinv.T)
    gen = eigh(S, Rinv, eigvals_only=True)
    return {
        "asym": np.linalg.norm(A - A.T) / np.linalg.norm(A),
        "lmin_S": np.linalg.eigvalsh(S)[0],
        "lmin_Rinv": np.linalg.eigvalsh(Rinv)[0],
        "gen_min": gen[0],
        "gen_max": gen[-1],
        "err_A": plain.metrics["rel_l2"],
        "err_S": sym.metrics["rel_l2"],
        "bytes_A": plain.metrics["bytes_apply"],
        "bytes_S": sym.metrics["bytes_apply"],
    }


def main():
    t0 = time.time()
    print("## alpha x h/l (uniform, 3000 obs, 4x3 tasks)\n")
    print("| alpha | h/l | asym | lmin(S) | lmin(R^-1) | gen eig min | gen eig max "
          "| err A~ | err S | bytes A~ | bytes S |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for alpha in (0.5, 0.8, 0.95):
        for hl in (1, 2, 3, 4, 5, 6):
            r = analyse(Config(alpha=alpha, h=hl * BASE.c))
            print(f"| {alpha} | {hl} | {r['asym']:.1e} | {r['lmin_S']:.3g} | {r['lmin_Rinv']:.3g} "
                  f"| {r['gen_min']:.4f} | {r['gen_max']:.4f} | {r['err_A']:.1e} | {r['err_S']:.1e} "
                  f"| {r['bytes_A'] / 1e3:.0f} kB | {r['bytes_S'] / 1e3:.0f} kB |", flush=True)

    print("\n## robustness (alpha = 0.95, the worst conditioned)\n")
    print("| tasks | pattern | h/l | asym | lmin(S) | gen eig min | gen eig max | err A~ | err S |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    cases = [((4, 3), p) for p in ("clustered", "coincident", "boundary", "geostationary")]
    cases += [((8, 6), "uniform"), ((8, 6), "clustered")]
    for (px, py), pattern in cases:
        for hl in (2, 4):
            r = analyse(Config(alpha=0.95, h=hl * BASE.c, px=px, py=py, pattern=pattern))
            print(f"| {px}x{py} | {pattern} | {hl} | {r['asym']:.1e} | {r['lmin_S']:.3g} "
                  f"| {r['gen_min']:.4f} | {r['gen_max']:.4f} | {r['err_A']:.1e} | {r['err_S']:.1e} |",
                  flush=True)
    print(f"\nTotal time: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
