"""Phase A step 5b: equal-area bands x sectors against density-balanced patches (RCB).

Realistic SEVIRI layout (pattern "geo_disc", RESULTS 2.9): l = c = 60 km (provisional),
alpha = 0.8, symmetrised operator. One patch per task (relaxed mode), so the patch
statistics are task statistics.

Cases
  one image   5 300 pixels at the calibrated 55 km spacing (a channel-6-sized image;
              accuracy depends on the local density, which is the same for every
              channel, RESULTS 2.9). The dense exact reference for 11 000 pixels needs
              several GB, too much for a login session.
  two discs   Meteosat-11 ch 5 (10 680 pixels, lon 0) + Meteosat-8 ch 5 (8 904 pixels,
              lon 41.5 E) as two groups: two satellites seen in one 15-min slot, as in
              the CCMA. Cost only (no exact reference).
Partitions
  bands/rect  equal-area SphereDecomposition, exact geometric halo ("rect")
  bands/obs   the same patches, halo "within h of a core observation" ("obs")
  bisection   BisectionDecomposition: equal observation counts per patch ("obs" halo)

Load imbalance is max/mean over ALL patches (a patch with no data is an idle task):
core size, and local cost sum_g n_loc^3 (setup) and sum_g n_loc^2 (per application).

    python experiments/partitions.py > experiments/partitions.out
"""
import numpy as np

from r_matrix import Config, run
from r_matrix.generate import make_obs

BASE = Config(geometry="sphere", pattern="geo_disc", c=60.0, alpha=0.8, symmetric=True)
PARTS = (("bands/rect", dict(partition="bands", criterion="rect")),
         ("bands/obs", dict(partition="bands", criterion="obs")),
         ("bisection", dict(partition="bisection", criterion="obs")))


def two_discs(P):
    """Met-11-like disc at lon 0 (group 0) and Met-8-like disc at 41.5 E (group 1)."""
    a = make_obs(10680, None, P, seed=0, pattern="geo_disc", geometry="sphere")
    b = make_obs(8904, None, P, seed=1, pattern="geo_disc", geometry="sphere")
    out = []
    for oa, ob in zip(a, b):
        ob = ob.copy()
        ob["x"] += np.radians(41.5)
        ob["global_id"] += 10_000_000
        ob["group"] = 1
        o = np.concatenate([oa, ob])
        o["orig_index"] = np.arange(len(o))
        out.append(o)
    return out


def stats(res):
    cores = np.array([s.ncore for s in res.solvers], dtype=float)
    blocks = [[idx.size for idx, _, _, _ in s.blocks] for s in res.solvers]
    c3 = np.array([sum(b ** 3 for b in bl) for bl in blocks], dtype=float)
    c2 = np.array([sum(b ** 2 for b in bl) for bl in blocks], dtype=float)
    m = res.metrics
    return (f"{np.count_nonzero(cores)} | {cores.max() / cores.mean():.1f} | {c3.max() / c3.mean():.1f} "
            f"| {c2.max() / c2.mean():.1f} | {m['local_n_max']} | {m['halo_core_ratio_peak']:.1f} "
            f"| {sum(8 * b * b for bl in blocks for b in bl) / 1e6:.0f} MB | {m['bytes_apply'] / 1e3:.0f} kB "
            f"| {m['messages_apply']}")


HEAD = ("| patches | partition | rel. L2 error | with data | max/mean core | max/mean n^3 | max/mean n^2 "
        "| largest local n | peak halo/core | sum of local R | bytes / apply | messages / apply |")
RULE = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"

for hl in (1, 2, 3, 4):
    print(f"## One image (5 300 pixels), h = {hl} l = {hl * BASE.c:.0f} km\n")
    print(HEAD)
    print(RULE)
    for px, py in ((8, 6), (16, 12)):
        for name, kw in PARTS:
            res = run(BASE, n_obs=5300, px=px, py=py, h=hl * BASE.c, **kw)
            print(f"| {px * py} | {name} | {res.metrics['rel_l2']:.1e} | {stats(res)} |", flush=True)
    print()

print("## Two discs (Met-11 + Met-8 ch 5, 19 584 pixels, 2 groups), cost only\n")
print("Same halo (4 l) for both, and the equal-accuracy pair from the one-image tables: bands/rect")
print("at 2 l and bisection at 3 l both meet a 1e-3 error gate.\n")
print(HEAD.replace("| partition |", "| partition | h/l |").replace("| rel. L2 error ", ""))
print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for px, py in ((8, 6), (16, 12)):
    obs = two_discs(px * py)
    for name, kw, hl in (("bands/rect", PARTS[0][1], 4), ("bands/obs", PARTS[1][1], 4),
                         ("bisection", PARTS[2][1], 4), ("bands/rect", PARTS[0][1], 2),
                         ("bisection", PARTS[2][1], 3)):
        res = run(BASE, obs_init=obs, px=px, py=py, h=hl * BASE.c, reference=False, **kw)
        print(f"| {px * py} | {name} | {hl} | {stats(res)} |", flush=True)
