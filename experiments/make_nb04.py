"""Build notebooks/04_vecchia_evaluation.ipynb (results of the Vecchia evaluation).

    python experiments/make_nb04.py notebooks/04_vecchia_evaluation.ipynb
"""
import sys

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md("""# 04 · Vecchia prototype: evaluation against core + halo

This notebook compares the two Vecchia variants of `03_vecchia_prototype.ipynb` with the core + halo operator now in rcorr, the symmetrised Ãₛ. It covers everything that matters operationally:

1. **Accuracy:** generalised eigenvalues of C̃⁻¹C, KL divergence, the error of Jₒ and of R⁻¹d for real departures.
2. **Symmetry and positive definiteness.**
3. **Implied covariance:** the variances and correlations that perturbations would have.
4. **Setup cost:** flops, the communication pattern, rounds.
5. **Apply cost:** flops, bytes, peers.
6. **Sampling:** rounds of the forward substitution.
7. **Behaviour from 96 to 1024 tasks.**

The notebook only reads results. They come from three scripts:

| results | script | where it runs |
|---|---|---|
| `golden.json` | `experiments/vecchia_golden.py` | login node, about 5 min |
| `acc_ch*_s*_a*_P*.json` | `experiments/vecchia_seviri.py <npz> <channel> <slot> <alpha> <P> <out>` | Slurm array, one image per task |
| `scaling.json` | `experiments/vecchia_scaling.py <npz> <out>` | Slurm, about 15 min |

The real data are all active Met-11 SEVIRI clear-sky radiances of the jdj6 cycle 2022-06-01 00 UTC, 203 633 data:
* ch5: 128k
* ch6: 64k
* ch9: 12k

Each group is one channel × one 15-min slot (36 groups). d = fg_depar / σ_o, c = ℓ = 115 km, and the GC support is 2ℓ = 230 km. The **patches are the RCB partition of all data of the cycle** into P patches, as rcorr does, so a single image has few data per patch at large P."""),
code("""import glob, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
RES = Path(os.environ.get("VECCHIA_RESULTS", Path(os.environ.get("HPCPERM", "~")) / "r-package/vecchia/results")).expanduser()
COL = {"core+halo": "#2a78d6", "block": "#eb6834", "point": "#1baf7a"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e5e5e3", "grid.linewidth": 0.6})
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
print(RES)"""),
md("""## 1. Golden cases

There are 9 cases: uniform, clustered, several groups, poles and dateline, the geostationary disc, two classes, inactive data, and a short halo. Data and parameters are those of `export.CASES`, with c of 400 to 1500 km, α of 0.8 to 0.9 and h of 1.5c to 4c. We use 8 RCB patches. Every operator uses the class's halo width h.
* **core + halo:** symmetrised.
* **block:** one-sided halo of the same width.
* **point:** m nearest earlier data."""),
code("""g = json.load(open(RES / "golden.json"))
d2 = pd.DataFrame([x for x in g if x["method"] == "d2"]).set_index("case")[["point", "block"]]
gd = pd.DataFrame([x for x in g if x["method"] != "d2"])
gd["variant"] = (gd.method + " " + gd.seq.fillna("") + gd.order.fillna("") + " " + gd.m.fillna(0).astype(int).astype(str)).str.replace(" 0$", "", regex=True).str.replace(r"\\s+", " ", regex=True).str.strip()
keep = ["core+halo", "block colour", "block id", "point maxmin 10", "point maxmin 20", "point maxmin 50", "point patch 20"]
t = gd[gd.P == 8].pivot_table(index=["case", "group", "n", "alpha", "h_over_c"], columns="variant", values="max_dev", aggfunc="first")[keep]
print("max |lambda - 1| (8 patches)")
display(t.style.format("{:.1e}"))
print("smallest eigenvalue over all cases and groups (> 0: SPD):")
display(gd.groupby("variant")["lam_min"].min().to_frame().T.style.format("{:.3f}"))
print("D2: C~^-1 x bit-identical on 1, 2, 4, 7 tasks (8 fixed patches)")
display(d2.T)"""),
code("""h = gd[gd.P == 8].pivot_table(index=["case", "group"], columns="variant", values="halo_over_core", aggfunc="first")[["core+halo", "block colour", "point maxmin 20", "point maxmin 50"]]
r = gd[gd.P == 8].pivot_table(index=["case", "group"], columns="variant", values="sample_rounds", aggfunc="first")[["block colour", "block id", "point maxmin 20", "point patch 20"]]
display(pd.concat({"halo / core": h, "sampling rounds": r}, axis=1).style.format("{:.2f}", subset=pd.IndexSlice[:, "halo / core"]).format("{:.0f}", subset=pd.IndexSlice[:, "sampling rounds"]))"""),
md("""**Golden cases, in short.**

* **Block Vecchia** is as accurate as the symmetrised core + halo at the same h on every case, often better. Examples: geo disc 3e-2 against 7e-2, clustered 3e-3 against 1e-2. It needs half the halo.
* **Point Vecchia** depends on how many data lie within the correlation length. Where the data are dense relative to c (clustered with α 0.9, geo disc), m ≤ 20 gives max |λ − 1| of about 1, i.e. λ_min of about 0.5. It is still SPD.
* **SPD and D2:** every variant is SPD everywhere, and all are bit-identical across task counts."""),
md("""## 2. One real image: accuracy, implied covariance, costs

Each `acc_*.json` holds one image, one α and one P, with these operators:
* **core + halo:** h = 2, 3, 4c.
* **block:** sequence `colour`, h = 2, 3, 4, 6c; sequence `id`, h = 4c.
* **point:** orderings `maxmin` and `patch`, m = 30, 50, 100.

**Columns:**

| column | meaning |
|---|---|
| `max_dev`, `lam_min`, `kl` | generalised eigenvalues of C̃⁻¹C |
| `jo_rel` | relative error of dᵀC̃⁻¹d for the real d |
| `z_rel` | relative error of C̃⁻¹d |
| `var_err_max` | largest error of the implied variances diag(C̃) (what EDA perturbations would have) |
| `cov_err_max` | largest error of any implied covariance |
| `flops_apply_max/total` | one C̃⁻¹x: on the busiest patch, and summed. For core + halo this is two right-hand sides through the Cholesky factor, as rcorr does today |
| `flops_apply_explicit_max` | core + halo with the core rows of R_loc⁻¹ stored explicitly: a cheaper apply that rcorr could adopt (not in all files) |
| `flops_setup_max` | the factorisation on the busiest patch |
| `bytes_apply`, `peers_max` | communication of one C̃⁻¹x |
| `bytes_setup`, `collectives_setup`, `rounds_setup` | communication of the setup |
| `sample_rounds` | sequential communication rounds of x = W⁻¹z |"""),
code("""rows = []
for f in sorted(glob.glob(str(RES / "acc_ch*_s*_a*_P*.json"))):
    r = json.load(open(f))
    for x in r["rows"]:
        rows.append(dict(channel=r["channel"], slot=r["slot"], alpha=r["alpha"], P=r["P"], n=r["n"], **x))
acc = pd.DataFrame(rows)
acc["param"] = acc.h_over_c.where(acc.method != "point", acc.m)
acc["variant"] = acc.method + np.where(acc.method == "block", " " + acc.seq.fillna(""), "") + np.where(acc.method == "point", " " + acc.order.fillna(""), "")
acc["label"] = acc.variant + np.where(acc.method == "point", " m=" + acc.m.fillna(0).astype(int).astype(str), " h=" + acc.h_over_c.astype(str) + "c")
cases = acc.groupby(["channel", "slot", "alpha", "P", "n"]).size().rename("operators").reset_index()
cases"""),
code("""for c_ in ("flops_apply_explicit_max", "flops_setup_explicit_max"):
    if c_ not in acc:
        acc[c_] = np.nan
cols = ["label", "max_dev", "lam_min", "kl", "jo_rel", "z_rel", "var_err_max", "cov_err_max", "halo_over_core", "peers_max",
        "flops_apply_max", "flops_apply_explicit_max", "flops_apply_total", "flops_setup_max", "bytes_apply", "bytes_setup", "rounds_setup", "sample_rounds"]
fmt = {c: "{:.1e}" for c in ["max_dev", "kl", "jo_rel", "z_rel", "var_err_max", "cov_err_max", "flops_apply_max", "flops_apply_explicit_max", "flops_apply_total", "flops_setup_max", "bytes_apply", "bytes_setup"]}
fmt.update(lam_min="{:.4f}", halo_over_core="{:.2f}", peers_max="{:.0f}", rounds_setup="{:.0f}", sample_rounds="{:.0f}")
def show(channel, slot, alpha, P):
    s = acc[(acc.channel == channel) & (acc.slot == slot) & (acc.alpha == alpha) & (acc.P == P)][cols]
    print(f"ch{channel} slot {slot}, alpha {alpha}, P = {P}, n = {acc[(acc.channel == channel) & (acc.slot == slot)].n.iloc[0]}")
    display(s.set_index("label").style.format(fmt, na_rep="-"))
show(6, 28, 0.5, 96)"""),
code("""show(6, 28, 0.7, 96)"""),
md("""### 2.1 Accuracy against communication and against apply cost

Every operator is one point, and a line joins one family as its parameter grows: h for core + halo and block, m for point.
* **Left:** accuracy against the bytes of one application.
* **Middle:** accuracy against the flops of the busiest patch.
* **Right:** accuracy against the setup flops of the busiest patch.

Lower left is better."""),
code("""def pareto(channel, slot, alpha, P, ax_row):
    s = acc[(acc.channel == channel) & (acc.slot == slot) & (acc.alpha == alpha) & (acc.P == P)]
    for (variant, grp) in s.groupby("variant"):
        grp = grp.sort_values("param")
        meth = grp.method.iloc[0]
        style = dict(color=COL[meth], lw=2, ms=6, marker="o" if ("colour" in variant or "maxmin" in variant or meth == "core+halo") else "s",
                     ls="-" if ("colour" in variant or "maxmin" in variant or meth == "core+halo") else ":")
        for ax, xc in zip(ax_row, ("bytes_apply", "flops_apply_max", "flops_setup_max")):
            ax.loglog(grp[xc], grp.max_dev, label=variant, **style)
            for _, q in grp.iterrows():
                if ax is ax_row[0]:
                    ax.annotate(f"{q.param:g}", (q[xc], q.max_dev), fontsize=7, xytext=(3, 3), textcoords="offset points", color="#52514e")
    ax_row[0].set_ylabel(f"max |λ−1|   (ch{channel}, α {alpha}, P {P})")
    for ax in ax_row:
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    for ax, t in zip(ax_row, ("bytes per application", "apply flops, busiest patch", "setup flops, busiest patch")):
        ax.set_xlabel(t)
fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), sharey="row")
pareto(6, 28, 0.5, 96, axs[0]); pareto(6, 28, 0.7, 96, axs[1])
axs[0, 0].legend(fontsize=8)
fig.suptitle("ch6 slot 28, 96 patches: accuracy against cost (labels: h/c or m)")
fig.tight_layout()"""),
md("""### 2.2 Dependence on α (the nugget)

The operators shown are core + halo at h = 4c, block (colour) at h = 4c and 6c, and point (max-min) at m = 50 and 100. α = 0.9 is beyond today's settings (jdr3 uses 0.7 for WV and 0.5 for IR), but it shows the trend."""),
code("""sel = {"core+halo h=4.0c": "core+halo", "block colour h=4.0c": "block", "block colour h=6.0c": "block", "point maxmin m=50": "point", "point maxmin m=100": "point"}
s = acc[(acc.channel == 6) & (acc.slot == 28) & (acc.P == 96) & acc.label.isin(sel)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for lab, grp in s.groupby("label"):
    grp = grp.sort_values("alpha")
    ls = ":" if ("6.0" in lab or "100" in lab) else "-"
    ax[0].semilogy(grp.alpha, grp.max_dev, marker="o", ls=ls, color=COL[sel[lab]], lw=2, label=lab)
    ax[1].semilogy(grp.alpha, grp.var_err_max, marker="o", ls=ls, color=COL[sel[lab]], lw=2, label=lab)
ax[0].set(xlabel="α", ylabel="max |λ − 1|", title="accuracy of C̃⁻¹"); ax[1].set(xlabel="α", ylabel="max |diag(C̃) − 1|", title="implied variances")
ax[0].legend(fontsize=8); fig.tight_layout()"""),
md("""### 2.3 Dependence on the number of patches

As P grows, the patches shrink, but the patch layout still comes from all data of the cycle. For ch6 at α 0.7 we compare 96, 384 and 1024 patches. Two quantities matter:
* the accuracy, which should depend on h and not on P;
* the costs per patch."""),
code("""s = acc[(acc.channel == 6) & (acc.slot == 28) & (acc.alpha == 0.7) & acc.label.isin(["core+halo h=4.0c", "block colour h=4.0c", "block colour h=6.0c", "block id h=4.0c", "point maxmin m=50", "point patch m=50"])]
display(s.pivot_table(index="label", columns="P", values=["max_dev", "halo_over_core", "flops_apply_max", "peers_max", "sample_rounds"]).style.format("{:.2g}"))"""),
code("""others = cases[~((cases.channel == 6) & (cases.slot == 28) & cases.alpha.isin([0.5, 0.7]) & (cases.P == 96))]
for _, c in others.iterrows():
    show(c.channel, c.slot, c.alpha, c.P)"""),
md("""## 3. All 36 groups of the cycle, 96 to 1024 tasks

These are structural counts from `vecchia_scaling.py`, without factorisation. Every patch holds all groups. Per task, summed over the groups:
* the halo,
* flops of one application,
* flops of the setup,
* bytes of one application (halo + adjoint halo),
* peers.

Sampling rounds and point-setup rounds are also counted. The block variant uses the colour sequence."""),
code("""sc = json.load(open(RES / "scaling.json"))
recs = []
for x in sc["results"]:
    for k in ("core_halo", "core_halo_explicit", "block_h3", "block_h4", "block_h4_id", "point_m50", "point_m100"):
        if k not in x:
            continue
        y = x[k]
        recs.append(dict(P=x["P"], op=k, diam_km=x["patch_diam_km"], colours=x["colours"], halo_over_core=y["halo_over_core"],
                         apply_max=y["flops_apply"]["max"], apply_mean=y["flops_apply"]["mean"],
                         setup_max=y["flops_setup"]["max"], setup_mean=y["flops_setup"]["mean"],
                         words_max=y["words"]["max"], bytes_max=y["bytes_apply"]["max"],
                         peers_max=y["peers"]["max"], peers_mean=y["peers"]["mean"],
                         sample_rounds=y.get("sample_rounds", np.nan), setup_rounds=y.get("setup_rounds", np.nan)))
S = pd.DataFrame(recs)
print(f"n = {sc['n']}, groups = {sc['n_groups']}")
display(S.pivot_table(index="op", columns="P", values=["halo_over_core", "apply_max", "setup_max", "bytes_max", "peers_max", "sample_rounds"]).style.format("{:.3g}"))"""),
code("""OPC = {"core_halo": ("core+halo h=4c", COL["core+halo"], "-"),
       "core_halo_explicit": ("core+halo h=4c, explicit rows", COL["core+halo"], "--"), "block_h4": ("block h=4c", COL["block"], "-"), "block_h3": ("block h=3c", COL["block"], ":"),
       "block_h4_id": ("block h=4c, id order", "#eda100", "--"),
       "point_m50": ("point m=50", COL["point"], ":"), "point_m100": ("point m=100", COL["point"], "-")}
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for op, (lab, col, ls) in OPC.items():
    s = S[S.op == op].sort_values("P")
    if s.empty:
        continue
    kw = dict(color=col, ls=ls, lw=2, marker="o", ms=5, label=lab)
    axs[0, 0].plot(s.P, s.halo_over_core, **kw)
    axs[0, 1].loglog(s.P, s.apply_max, **kw)
    axs[0, 2].plot(s.P, s.apply_max.iloc[0] / s.apply_max, **kw)
    axs[1, 0].loglog(s.P, s.setup_max, **kw)
    axs[1, 1].loglog(s.P, s.peers_max, **kw)
    if not op.startswith("core_halo"):
        axs[1, 2].semilogy(s.P, s.sample_rounds, **kw)
axs[0, 2].plot([96, 1024], [1, 1024 / 96], color="#c3c2b7", lw=1, label="ideal")
for ax, t in zip(axs.flat, ("halo / core", "apply flops, busiest task", "apply speed-up of the busiest task vs 96",
                            "setup flops, busiest task", "peers of the busiest task", "sampling rounds (W⁻¹z)")):
    ax.set_title(t, fontsize=10); ax.set_xlabel("tasks (= patches)")
axs[0, 0].set_yscale("log"); axs[0, 2].legend(fontsize=8)
fig.tight_layout()"""),
code("""pm = pd.DataFrame([dict(P=x["P"], m=m, **x[f"point_m{m}"]["reach_km"], setup_rounds=x[f"point_m{m}"]["setup_rounds"],
                        **{f"certified_after_{k}": v for k, v in x[f"point_m{m}"]["setup_rounds_frac"].items()})
                   for x in sc["results"] for m in (50, 100)])
print("point Vecchia: reach of the conditioning sets (km; c = 115 km, today's halo 460 km), and the fraction of data whose")
print("neighbour search is certified after k rounds of radius doubling from R0 = 4c = 460 km")
display(pm[pm.P == 96].drop(columns="P").set_index("m").style.format("{:.3g}"))"""),
md("""## 4. Summary

See `VECCHIA_REPORT.md` at the top of the repository for the conclusions and recommendations. The numbers quoted there are the ones in this notebook."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, sys.argv[1])
