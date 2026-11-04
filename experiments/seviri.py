"""Phase A step 5: realistic SEVIRI images (pattern "geo_disc" on the sphere).

The layout is calibrated on the 2022060100 CCMA of experiment jdfp (plan repository,
plan/ifs_deep_dive/05_seviri_ccma_quicklook.md):
- pixels about 55 km apart (median nearest-neighbour distance), the same for every
  channel; channels with fewer active pixels have more of the disc masked;
- active pixels per hourly image: Meteosat-11 ch 5 / 6 / 9: 10 680 / 5 297 / 991;
  Meteosat-8 ch 5 / 6: 8 904 / 4 939. 12 images per 12-hour window, so 60 groups.

The error parameters are PROVISIONAL: l = c = 60 km (first-guess departures of the
water-vapour channels correlate 0.56 below 50 km, 0.3 at 50-100 km and about 0 beyond
150 km) and alpha in {0.5, 0.8}. They are to be replaced by proper diagnostics (plan
Phase B step 6). Part C brackets l up to 150 km.

All runs: symmetrised operator, global equal-area SphereDecomposition. Accuracy is
given as the median and range over three seeds (cloud pattern, jitter, departures).

A  accuracy against h/l, channel-6-sized image (5300 pixels), 192 patches
B  cost of one channel-5-sized image (11000 pixels) for 48 / 192 / 768 patches
C  sensitivity to l and image size, alpha = 0.8, 192 patches
D  totals for one 12-hour window (60 groups), 192 patches, l = 60 km, h = 4 l

    python experiments/seviri.py > experiments/seviri.out
"""
import numpy as np

from r_matrix import Config, run
from r_matrix.diagnostics import load_balance

BASE = Config(geometry="sphere", pattern="geo_disc", c=60.0, symmetric=True)
SEEDS = (0, 1, 2)


def err(**kw):
    """Median, min and max of the relative L2 error over SEEDS, plus the last metrics."""
    e, m = [], None
    for s in SEEDS:
        m = run(BASE, seed=s, **kw).metrics
        e.append(m["rel_l2"])
    return np.median(e), min(e), max(e), m


def cost(**kw):
    res = run(BASE, departures="white", reference=False, **kw)
    blocks = [idx.size for s in res.solvers for idx, _, _, _ in s.blocks]
    cores = np.array([s.ncore for s in res.solvers])
    return {
        "with_data": int(np.count_nonzero(cores)),
        "core_max": int(cores.max()),
        "n_max": res.metrics["local_n_max"],
        "R_max": res.metrics["local_R_bytes_max"],
        "R_sum": sum(8 * b * b for b in blocks),
        "flop_sum": sum(b ** 3 / 3 for b in blocks),
        "cv_data": load_balance(cores[cores > 0])["cv"],
        "bytes": res.metrics["bytes_apply"],
        "setup": res.metrics["time_setup"],
    }


print("## A. accuracy, channel-6-sized image (5300 pixels, 192 patches)\n")
print("| alpha | h/l | h (km) | rel. L2 error: median | range (3 seeds) | peak halo/core | largest local n |")
print("|---:|---:|---:|---:|---|---:|---:|")
for alpha in (0.5, 0.8):
    for hl in (2, 3, 4, 5, 6):
        med, lo, hi, m = err(n_obs=5300, px=16, py=12, alpha=alpha, h=hl * BASE.c)
        print(f"| {alpha} | {hl} | {hl * BASE.c:.0f} | {med:.1e} | {lo:.1e} – {hi:.1e} "
              f"| {m['halo_core_ratio_peak']:.1f} | {m['local_n_max']} |", flush=True)

print("\n## B. cost of one channel-5-sized image (11000 pixels, alpha = 0.8, h = 4 l = 240 km)\n")
print("| patches | with data | largest core | CV of core sizes (patches with data) | largest local n "
      "| largest local R | sum of local R | sum of n^3/3 | bytes / apply | setup time |")
print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for px, py in ((8, 6), (16, 12), (32, 24)):
    c = cost(n_obs=11000, px=px, py=py, alpha=0.8, h=4 * BASE.c)
    print(f"| {px * py} | {c['with_data']} | {c['core_max']} | {c['cv_data']:.2f} | {c['n_max']} "
          f"| {c['R_max'] / 1e6:.1f} MB | {c['R_sum'] / 1e6:.0f} MB | {c['flop_sum'] / 1e9:.2f} Gflop "
          f"| {c['bytes'] / 1e3:.0f} kB | {c['setup']:.1f} s |", flush=True)

print("\n## C. sensitivity (alpha = 0.8, 192 patches)\n")
print("| pixels | l (km) | h/l | rel. L2 error: median | range (3 seeds) | peak halo/core | largest local n |")
print("|---:|---:|---:|---:|---|---:|---:|")
for n, c in ((11000, 60.0), (5300, 100.0), (11000, 100.0), (5300, 150.0), (11000, 150.0)):
    for hl in (2, 3, 4):
        med, lo, hi, m = err(n_obs=n, c=c, px=16, py=12, alpha=0.8, h=hl * c)
        print(f"| {n} | {c:.0f} | {hl} | {med:.1e} | {lo:.1e} – {hi:.1e} "
              f"| {m['halo_core_ratio_peak']:.1f} | {m['local_n_max']} |", flush=True)

print("\n## D. one 12-hour window: 12 images x 5 channel groups (192 patches, l = 60 km, h = 4 l)\n")
print("| group | pixels | sum of local R | sum of n^3/3 | x 12 images: R | x 12: flop |")
print("|---|---:|---:|---:|---:|---:|")
tot_R = tot_f = 0.0
for name, n in (("Met-11 ch5", 10680), ("Met-11 ch6", 5297), ("Met-11 ch9", 991),
                ("Met-8 ch5", 8904), ("Met-8 ch6", 4939)):
    c = cost(n_obs=n, px=16, py=12, alpha=0.8, h=4 * BASE.c)
    tot_R += 12 * c["R_sum"]
    tot_f += 12 * c["flop_sum"]
    print(f"| {name} | {n} | {c['R_sum'] / 1e6:.0f} MB | {c['flop_sum'] / 1e9:.2f} Gflop "
          f"| {12 * c['R_sum'] / 1e9:.2f} GB | {12 * c['flop_sum'] / 1e9:.1f} Gflop |", flush=True)
print(f"| **window total** | {12 * (10680 + 5297 + 991 + 8904 + 4939)} | | "
      f"| **{tot_R / 1e9:.2f} GB** | **{tot_f / 1e9:.0f} Gflop** |")
