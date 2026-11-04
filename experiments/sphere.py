"""Phase A step 3: the core + halo approximation on the sphere.

3000 obs over the whole sphere with l = c = 1000 km (about 5.9 obs per l^2, close to the
plane baseline's 6.9), equal-area 4 x 3 and 8 x 6 SphereDecompositions, the
symmetrised operator and alpha = 0.8. The plane baseline (4 x 3) is shown for
comparison. "poles_dateline" puts 20 % of the points on 180 E/W, at or near the poles,
or with longitude stored in [pi, 3 pi).

    python experiments/sphere.py > experiments/sphere.out
"""
from r_matrix import Config, run

PLANE = Config(symmetric=True)
SPHERE = Config(geometry="sphere", c=1000.0, symmetric=True)

print("| geometry | tasks | pattern | h/l | rel. L2 error | residual distr. = global "
      "| peak halo/core | largest local n | bytes / apply |")
print("|---|---|---|---:|---:|---|---:|---:|---:|")
cases = [("plane", PLANE, 4, 3, "uniform"), ("sphere", SPHERE, 4, 3, "uniform"),
         ("sphere", SPHERE, 4, 3, "poles_dateline"), ("sphere", SPHERE, 8, 6, "uniform")]
for name, cfg, px, py, pattern in cases:
    for hl in (1, 2, 3, 4, 5, 6):
        m = run(cfg, px=px, py=py, pattern=pattern, h=hl * cfg.c).metrics
        same = abs(m["residual_distributed"] - m["residual_global"]) <= 1e-10 * m["residual_global"]
        print(f"| {name} | {px}x{py} | {pattern} | {hl} | {m['rel_l2']:.1e} "
              f"| {'yes' if same else 'no'}{' (h < 2l)' if hl < 2 else ''} "
              f"| {m['halo_core_ratio_peak']:.1f} | {m['local_n_max']} | {m['bytes_apply'] / 1e3:.0f} kB |",
              flush=True)
