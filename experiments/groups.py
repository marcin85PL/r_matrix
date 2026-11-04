"""Phase A step 2: correlation groups (e.g. channel x time slot).

The total number of observations is fixed at 3000. They are split into n_groups groups,
either "colocated" (each position carries one observation per group, like the channels of
a pixel) or "random" (independent positions per group). Each rank factorises one block
per group, so the cost is sum_g n_g^3 instead of (sum_g n_g)^3. Symmetrised operator,
alpha = 0.8, h = 4 l (h = 2.0 with l = c = 0.5).

    python experiments/groups.py > experiments/groups.out
"""
from r_matrix import Config, run

print("| n_groups | layout | rel. L2 error | largest local n (all groups) "
      "| largest local R memory | bytes / apply | setup time |")
print("|---:|---|---:|---:|---:|---:|---:|")
for layout in ("colocated", "random"):
    for G in (1, 2, 4, 8):
        if G == 1 and layout == "random":
            continue
        m = run(Config(n_groups=G, group_layout=layout, h=2.0, symmetric=True)).metrics
        print(f"| {G} | {layout if G > 1 else '-'} | {m['rel_l2']:.1e} | {m['local_n_max']} "
              f"| {m['local_R_bytes_max'] / 1e6:.2f} MB | {m['bytes_apply'] / 1e3:.0f} kB "
              f"| {m['time_setup']:.2f} s |", flush=True)
