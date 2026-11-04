"""Golden-data exporter: small deterministic cases for testing a compiled implementation.

    python -m r_matrix.export <case|all> <outdir>      # writes <outdir>/<case>/
    python -m r_matrix.export --list

Each case is a directory with ``manifest.json`` and one raw little-endian ``.bin`` file per
array (C order, ``float64`` or ``int64``, no header), so that a C++ reader needs no library.
Names are dotted (``obs.lon``) and the file is ``<name>.bin``.

The cases use the sphere with sigma = 1 (so R is the correlation matrix C, which is what the
library works with), Gaspari-Cohn on the chord distance, the ``obs`` halo criterion and
density-balanced (bisection) patches. See ``export_case`` for everything that is written and
``CASES`` for the case list.

Ragged data (per-patch lists) is stored CSR style: a concatenated array plus ``<name>_off``
with n_patches + 1 offsets.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from .covariance import build_R, gaspari_cohn
from .decomposition import SphereDecomposition
from .generate import make_obs
from .geometry import EARTH_RADIUS, points
from .pipeline import Config, patch_task_map, run

FORMAT = "rcorr-golden-1"
RELAXED_PATCHES = (1, 2, 4, 7)      # relaxed mode: one patch per task, tasks = patches
FIXED_PATCHES = 8                   # reproducible mode: fixed patches, any number of tasks
FIXED_TASKS = (1, 2, 4, 7)
BANDS = ((2, 2), (4, 3), (7, 1))    # SphereDecomposition shapes for the C3 owners

# name -> generator settings. c and h are in km. h = 2.5 c >= 2 c, so mult is exact.
CASES = {
    "tiny": dict(n=24, pattern="uniform", seed=11, c=1500.0, alpha=0.8, h=3750.0),
    "sph_uniform": dict(n=400, pattern="uniform", seed=1, c=1000.0, alpha=0.8, h=2500.0),
    "sph_clustered": dict(n=300, pattern="clustered", seed=2, c=1000.0, alpha=0.9, h=2500.0),
    "sph_groups": dict(n=360, pattern="uniform", seed=3, c=1000.0, alpha=0.8, h=2500.0,
                       n_groups=3, group_layout="colocated"),
    "sph_poles_dateline": dict(n=300, pattern="poles_dateline", seed=4, c=1500.0, alpha=0.8,
                               h=3750.0),
    "sph_geo_disc": dict(n=900, pattern="geo_disc", seed=5, c=400.0, alpha=0.8, h=1000.0),
}


def toy_version():
    """(git commit of the toy, dirty flag); ('unknown', False) outside a git checkout."""
    here = Path(__file__).resolve().parent

    def git(*a):
        return subprocess.run(["git", "-C", str(here), *a], capture_output=True, text=True,
                              check=True).stdout.strip()
    try:
        return git("rev-parse", "HEAD"), bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.CalledProcessError):
        return "unknown", False


def _csr(lists, dtype):
    """Concatenate a list of 1-D arrays; return (values, offsets)."""
    off = np.concatenate([[0], np.cumsum([len(a) for a in lists])]).astype(np.int64)
    vals = np.concatenate(lists) if lists else np.zeros(0)
    return vals.astype(dtype), off


def _block(prefix, ids, obs, c, alpha, out):
    """Chord distances and correlation block for the observations with global IDs `ids`."""
    pos = np.searchsorted(obs["global_id"][np.argsort(obs["global_id"])], ids)
    idx = np.argsort(obs["global_id"])[pos]
    o = obs[idx]
    xyz = points(o["x"], o["y"], "sphere")
    chord = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    out[f"{prefix}.ids"] = ids.astype(np.int64)
    out[f"{prefix}.chord"] = chord
    out[f"{prefix}.corr"] = build_R(o["x"], o["y"], np.ones(len(o)), c, alpha, o["group"], "sphere")


def _layout(obs, p, key, P, sym_run, plain_run, x, out):
    """Per-layout outputs: partition, owned and halo ID sets, local solves, global operator."""
    pre = f"layout.{key}."
    res = sym_run
    owned_ids = [o["global_id"] for o in res.owned]
    halo_ids = [h["global_id"] for h in res.halos]
    out[pre + "owned_ids"], out[pre + "owned_ids_off"] = _csr(owned_ids, np.int64)
    out[pre + "halo_ids"], out[pre + "halo_ids_off"] = _csr(halo_ids, np.int64)
    out[pre + "halo_owner"], _ = _csr(res.halo_owner, np.int64)
    out[pre + "halo_counts"] = np.array([np.bincount(ho, minlength=P) for ho in res.halo_owner],
                                        dtype=np.int64).reshape(P, P)
    owner_of = {int(g): q for q, ids in enumerate(owned_ids) for g in ids}
    out[pre + "owner"] = np.array([owner_of[int(g)] for g in obs["global_id"]], dtype=np.int64)
    # C2: the two local contributions to the symmetrised operator, per patch
    zc, wc, wh = [], [], []
    for s, o, h in zip(res.solvers, res.owned, res.halos):
        a, b, c_ = s.apply_sym_parts(o["value"], h["value"])
        zc.append(a); wc.append(b); wh.append(c_)
    out[pre + "z_core"], _ = _csr(zc, np.float64)
    out[pre + "w_core"], _ = _csr(wc, np.float64)
    out[pre + "w_halo"], _ = _csr(wh, np.float64)
    # C5: the global operator (obs order). inv_mult uses the symmetrised operator.
    out[pre + "zs_d"] = sym_run.z_orig[0]
    out[pre + "zs_x"] = sym_run.apply_Rinv([x])[0][0]
    out[pre + "zp_d"] = plain_run.z_orig[0]
    if key.startswith("fixed"):
        for t in FIXED_TASKS:
            out[pre + f"task_of_patch_t{t}"] = patch_task_map(P, t, "block")[0].astype(np.int64)


def export_case(name):
    """Build all arrays of one case. Returns (params, arrays) with arrays an ordered dict."""
    s = CASES[name]
    c, alpha, h = s["c"], s["alpha"], s["h"]
    groups = s.get("n_groups", 1)
    obs = make_obs(s["n"], None, 1, seed=s["seed"], pattern=s["pattern"], departures="correlated",
                   c=c, alpha=alpha, sigma_range=(1.0, 1.0), n_groups=groups,
                   group_layout=s.get("group_layout", "random"), geometry="sphere")[0]
    n = len(obs)
    rng = np.random.default_rng(1000 + s["seed"])
    x = rng.standard_normal(n)
    out = {}
    out["obs.global_id"] = obs["global_id"].astype(np.int64)
    out["obs.lon"], out["obs.lat"] = obs["x"].copy(), obs["y"].copy()
    out["obs.sigma"] = obs["sigma"].copy()
    out["obs.d"] = obs["value"].copy()
    out["obs.group"] = obs["group"].astype(np.int64)
    out["vec.x"] = x

    # ---- C1: geometry, GC, correlation blocks ----------------------------------------
    out["c1.xyz"] = points(obs["x"], obs["y"], "sphere")
    r = np.concatenate([[0.0, 0.5 * c, c, 1.5 * c, 2.0 * c - 1e-9, 2.0 * c, 2.5 * c],
                        np.linspace(0.0, 2.2 * c, 45)])
    out["c1.gc_r"], out["c1.gc_val"] = r, gaspari_cohn(r, c)
    ids = obs["global_id"]
    lrng = np.random.default_rng(2000 + s["seed"])
    _block("c1.listA", ids[:min(n, 40)], obs, c, alpha, out)
    _block("c1.listB", ids[lrng.permutation(n)[:min(n, 50)]], obs, c, alpha, out)

    # ---- C3: equal-area bands (bisection owners are in the layouts) ---------------------
    for px, py in BANDS:
        out[f"c3.owner_bands_{px}x{py}"] = SphereDecomposition(px, py).owner(
            obs["x"], obs["y"]).astype(np.int64)

    # ---- layouts: C2, C3 (bisection), C4, C5 ------------------------------------------
    base = Config(n_obs=n, py=1, geometry="sphere", partition="bisection", criterion="obs",
                  halo_method="kdtree", c=c, alpha=alpha, h=h, n_tasks=1)
    layouts = [(f"relaxed_p{P}", P) for P in RELAXED_PATCHES] + [(f"fixed_p{FIXED_PATCHES}",
                                                                  FIXED_PATCHES)]
    for key, P in layouts:
        runs = {sym: run(base, obs_init=[obs.copy()], px=P, symmetric=sym) for sym in (True, False)}
        _layout(obs, s, key, P, runs[True], runs[False], x, out)

    # ---- C5: exact global solve and exact matvec (independent of the layout) ---------------
    ref = runs[True]
    out["c5.exact_d"] = ref.apply_Rinv_exact([obs["value"]])[0]
    out["c5.exact_x"] = ref.apply_Rinv_exact([x])[0]
    R = build_R(obs["x"], obs["y"], obs["sigma"], c, alpha, obs["group"], "sphere")
    out["c5.mult_d"], out["c5.mult_x"] = R @ obs["value"], R @ x

    params = dict(case=name, geometry="sphere", earth_radius_km=EARTH_RADIUS, n_obs=n,
                  pattern=s["pattern"], seed=s["seed"], n_groups=groups,
                  group_layout=s.get("group_layout", "random"),
                  correlation="gaspari_cohn_chord", c_km=c, support_km=2 * c, alpha=alpha,
                  halo_km=h, halo_over_c=h / c, halo_criterion="obs", sigma="1 (R = C)",
                  partition="bisection", relaxed_patches=list(RELAXED_PATCHES),
                  fixed_patches=FIXED_PATCHES, fixed_task_counts=list(FIXED_TASKS),
                  bands=[list(b) for b in BANDS], patch_task_map="block")
    return params, out


def write_case(name, outdir):
    """Write one case to <outdir>/<name>/ and return that directory."""
    params, arrays = export_case(name)
    d = Path(outdir) / name
    d.mkdir(parents=True, exist_ok=True)
    commit, dirty = toy_version()
    entries = []
    for key, a in arrays.items():
        a = np.asarray(a)
        dt = "int64" if np.issubdtype(a.dtype, np.integer) else "float64"
        a = np.ascontiguousarray(a, dtype="<i8" if dt == "int64" else "<f8")
        (d / f"{key}.bin").write_bytes(a.tobytes())
        entries.append(dict(name=key, dtype=dt, shape=list(a.shape), file=f"{key}.bin"))
    manifest = dict(format=FORMAT, case=name, toy_commit=commit, toy_dirty=dirty,
                    numpy=np.__version__, params=params, arrays=entries)
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return d


def read_case(path):
    """Read a case directory. Returns (manifest, {name: ndarray}). For tests and checks."""
    d = Path(path)
    man = json.loads((d / "manifest.json").read_text())
    arrays = {}
    for e in man["arrays"]:
        dt = {"int64": "<i8", "float64": "<f8"}[e["dtype"]]
        arrays[e["name"]] = np.fromfile(d / e["file"], dtype=dt).reshape(e["shape"])
    return man, arrays


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m r_matrix.export", description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?", help=f"one of {', '.join(CASES)} or 'all'")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--list", action="store_true", help="list the cases")
    a = ap.parse_args(argv)
    if a.list:
        for k, v in CASES.items():
            print(k, v)
        return 0
    if not a.case or not a.outdir or (a.case != "all" and a.case not in CASES):
        ap.error(f"need <case> in {list(CASES)} or 'all', and <outdir>")
    for name in (CASES if a.case == "all" else [a.case]):
        print(write_case(name, a.outdir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
