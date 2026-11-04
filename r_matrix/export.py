"""Golden-data exporter: small deterministic cases for testing a compiled implementation.

    python -m r_matrix.export <case|all> <outdir> [--allow-dirty]    # writes <outdir>/<case>/
    python -m r_matrix.export --list

Each case is a directory with ``manifest.json`` and one raw little-endian ``.bin`` file per
array (C order, ``float64`` or ``int64``, no header), so that a C++ reader needs no library.
Names are dotted (``obs.lon``) and the file is ``<name>.bin``. The export refuses to run when
the toy checkout has uncommitted changes (tracked files, or untracked ``.py`` files) unless
``--allow-dirty`` is given, so that ``toy_commit`` in the manifest identifies the code.

All cases use the sphere with sigma = 1 (so R is the correlation matrix C, which is what the
library works with), Gaspari-Cohn on the chord distance, the ``obs`` halo criterion and
density-balanced (bisection) patches. A case has one or more *classes* (each with its own
length scale c, correlated fraction alpha and halo h); every group belongs to one class and
correlations act only inside a group. Data with ``obs.active == 0`` take no part: they are
not partitioned, have no halo, and pass through every operator unchanged (y = x).
Ragged per-patch data is CSR: a concatenated array plus ``<name>_off`` with P + 1 offsets.
Per-class arrays are named ``layout.<layout>.cls<k>.<array>`` and are aligned with that
class's ``owned_ids`` or ``halo_ids``; per-datum arrays (in ``obs.*`` order) are named
``layout.<layout>.<array>``. See ``export_case`` and ``CONVENTIONS``.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy

from .comm import MockComm
from .commmap import build_comm_maps, halo_exchange, halo_exchange_adjoint
from .covariance import LocalSolver, build_R, gaspari_cohn
from .decomposition import BisectionDecomposition, SphereDecomposition
from .generate import make_obs
from .geometry import EARTH_RADIUS, points
from .halo import discover_halos
from .pipeline import _to_patches, patch_task_map
from .redistribute import redistribute

FORMAT = "rcorr-golden-1"
RELAXED_PATCHES = (1, 2, 4, 7)      # relaxed mode: one patch per task, tasks = patches
FIXED_PATCHES = 8                   # reproducible mode: fixed patches, any number of tasks
FIXED_TASKS = (1, 2, 4, 7)
BANDS = ((2, 2), (4, 3), (7, 1))    # SphereDecomposition shapes (px, py) for the C3 owners

CONVENTIONS = dict(
    patch_numbering="0-based; halo_owner and owner hold patch ids",
    bisection_split=("node with k patches: split the points at the count quantile k1/k with "
                     "k1 = k // 2 (left gets k1 patches, right k - k1), along the axis of largest "
                     "extent of the rotated 3-D coordinates, cut moved to the nearest boundary "
                     "between distinct coordinate values (ties by global ID); patches are numbered "
                     "left to right in depth-first order"),
    band_index="owner = i + px * j; i = longitude sector of wrapped lon in [-pi, pi) (width 2 pi / px), "
               "j = equal-area latitude band in sin(lat) from the south pole",
    csr="per-patch arrays are concatenated in patch order; <name>_off holds P + 1 offsets",
    owned_order="owned_ids sorted by global ID; halo_ids sorted by (halo_owner, global ID)",
    group_order="CSR arrays are per datum (no group axis); local solves treat groups in ascending "
                "group id; groups without core data in a patch contribute 0 to its z, w",
    inactive="obs.active == 0: no patch, no halo, operators pass the value through (y = x); "
             "layout owner = -1",
    mult="layout mult_* is the halo matvec (exact only if halo >= 2 c); c5.mult_* is the exact C x",
    symmetrised="zs = 0.5 (z_core + w_core + adj_back); adj_back is w_halo summed into the owners",
)


def _case(n, pattern, seed, classes, **kw):
    return dict(n=n, pattern=pattern, seed=seed, classes=classes, **kw)


def _cls(c, alpha, h_over_c=2.5):
    return dict(c=c, alpha=alpha, h=h_over_c * c)


# c and h in km. h = 2.5 c >= 2 c except in sph_short_halo.
CASES = {
    "tiny": _case(24, "uniform", 11, [_cls(1500.0, 0.8)]),
    "sph_uniform": _case(400, "uniform", 1, [_cls(1000.0, 0.8)]),
    "sph_clustered": _case(300, "clustered", 2, [_cls(1000.0, 0.9)]),
    "sph_groups": _case(360, "uniform", 3, [_cls(1000.0, 0.8)], n_groups=3, group_layout="colocated"),
    "sph_poles_dateline": _case(300, "poles_dateline", 4, [_cls(1500.0, 0.8)]),
    "sph_geo_disc": _case(900, "geo_disc", 5, [_cls(400.0, 0.8)]),
    # two classes: groups 0, 2 -> class 0, groups 1, 3 -> class 1 (different c, alpha, h)
    "sph_classes": _case(240, "uniform", 6, [_cls(1000.0, 0.8), _cls(600.0, 0.9, 4.0)],
                         n_groups=4, group_layout="random", class_of_group=[0, 1, 0, 1]),
    # 20 % of the data inactive
    "sph_inactive": _case(240, "uniform", 7, [_cls(1000.0, 0.8)], inactive=0.2),
    # halo 1.5 c < 2 c: mult is approximate
    "sph_short_halo": _case(240, "uniform", 8, [_cls(1000.0, 0.8, 1.5)]),
}


def toy_version():
    """(git commit of the toy, dirty flag); ('unknown', True) outside a git checkout.

    Dirty means: changes in tracked files, or untracked .py files.
    """
    here = Path(__file__).resolve().parent

    def git(*a):
        return subprocess.run(["git", "-C", str(here), *a], capture_output=True, text=True,
                              check=True).stdout

    try:
        commit = git("rev-parse", "HEAD").strip()
        lines = git("status", "--porcelain", "--untracked-files=all").splitlines()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True
    dirty = any(not ln.startswith("??") or ln.endswith(".py") for ln in lines)
    return commit, dirty


def _csr(lists, dtype):
    """Concatenate a list of 1-D arrays; return (values, offsets)."""
    off = np.concatenate([[0], np.cumsum([len(a) for a in lists])]).astype(np.int64)
    vals = np.concatenate(lists) if lists else np.zeros(0)
    return vals.astype(dtype), off


def _corr_block(o, cls, classes):
    """alpha GC(chord) + (1 - alpha) I within groups, with each datum's class parameters.
    Data of different classes are never correlated (they are in different groups)."""
    n = len(o)
    out = np.zeros((n, n))
    for k, cl in enumerate(classes):
        m = np.flatnonzero(cls == k)
        if len(m):
            out[np.ix_(m, m)] = build_R(o["x"][m], o["y"][m], np.ones(len(m)), cl["c"], cl["alpha"],
                                        o["group"][m], "sphere")
    return out


def _block(prefix, ids, obs, cls, classes, out):
    """Chord distances and correlation block for the observations with global IDs `ids`."""
    order = np.argsort(obs["global_id"])
    idx = order[np.searchsorted(obs["global_id"][order], ids)]
    o = obs[idx]
    xyz = points(o["x"], o["y"], "sphere")
    out[f"{prefix}.ids"] = ids.astype(np.int64)
    out[f"{prefix}.chord"] = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    out[f"{prefix}.corr"] = _corr_block(o, cls[idx], classes)


def _layout(obs, cls, active, classes, P, key, vecs, out):
    """One partition into P patches: owners, per-class ID sets, local solves, global operators.

    vecs: {"d": ..., "x": ...}, full-length input vectors. Everything is computed from the toy's
    building blocks (redistribute, halo discovery, comm maps, LocalSolver) with one shared
    bisection of all active data, and per-class halos and solvers.
    """
    pre = f"layout.{key}."
    act = np.flatnonzero(active)
    a = obs[act]
    decomp = BisectionDecomposition.from_points(a["x"], a["y"], a["global_id"], P)
    task_of, home = patch_task_map(P, 1, "block")
    comm = MockComm(P, task_of)
    owner = np.full(len(obs), -1, dtype=np.int64)
    owner[act] = decomp.owner(a["x"], a["y"])
    out[pre + "owner"] = owner
    res = {f"{op}_{v}": vecs[v].copy() for op in ("zs", "zp", "mult") for v in vecs}
    for k, cl in enumerate(classes):
        pos = act[cls[act] == k]
        ok = obs[pos].copy()
        ok["orig_rank"], ok["orig_index"] = 0, np.arange(len(ok))
        owned, _, plan = redistribute(comm, decomp, _to_patches([ok], home, P))
        halos, halo_owner, _ = discover_halos(comm, decomp, owned, cl["h"], "obs", "kdtree")
        maps = build_comm_maps(comm, owned, halos, halo_owner)
        solvers = [LocalSolver(o, h, cl["c"], cl["alpha"], "sphere") for o, h in zip(owned, halos)]
        ncore = [s.ncore for s in solvers]
        cp = f"{pre}cls{k}."
        out[cp + "owned_ids"], out[cp + "owned_ids_off"] = _csr([o["global_id"] for o in owned], np.int64)
        out[cp + "halo_ids"], out[cp + "halo_ids_off"] = _csr([h["global_id"] for h in halos], np.int64)
        out[cp + "halo_owner"], _ = _csr(halo_owner, np.int64)
        out[cp + "halo_counts"] = np.array([np.bincount(ho, minlength=P) for ho in halo_owner],
                                           dtype=np.int64).reshape(P, P)
        for v, vec in vecs.items():
            d_core = plan.forward(comm, _to_patches([vec[pos]], home, P))
            d_halo = halo_exchange(comm, maps, d_core)
            parts = [s.apply_sym_parts(dc, dh) for s, dc, dh in zip(solvers, d_core, d_halo)]
            back = halo_exchange_adjoint(comm, maps, [p[2] for p in parts], ncore)
            for i, nm in enumerate(("z_core", "w_core", "w_halo")):
                out[f"{cp}{nm}_{v}"], _ = _csr([p[i] for p in parts], np.float64)
            out[f"{cp}adj_back_{v}"], _ = _csr(back, np.float64)
            z_sym = [0.5 * (p[0] + p[1] + b) for p, b in zip(parts, back)]
            z_plain = [s.apply(dc, dh) for s, dc, dh in zip(solvers, d_core, d_halo)]
            z_mult = [s.matvec_core(dc, dh) for s, dc, dh in zip(solvers, d_core, d_halo)]
            for nm, own in (("zs", z_sym), ("zp", z_plain), ("mult", z_mult)):
                res[f"{nm}_{v}"][pos] = plan.reverse(comm, own)[home[0]]
    for nm, arr in res.items():
        out[pre + nm] = arr
    if key.startswith("fixed"):
        for t in FIXED_TASKS:
            out[pre + f"task_of_patch_t{t}"] = patch_task_map(P, t, "block")[0].astype(np.int64)


def export_case(name):
    """Build all arrays of one case. Returns (params, arrays) with arrays an ordered dict."""
    s = CASES[name]
    classes = s["classes"]
    groups = s.get("n_groups", 1)
    obs = make_obs(s["n"], None, 1, seed=s["seed"], pattern=s["pattern"], departures="correlated",
                   c=classes[0]["c"], alpha=classes[0]["alpha"], sigma_range=(1.0, 1.0),
                   n_groups=groups, group_layout=s.get("group_layout", "random"),
                   geometry="sphere")[0]
    n = len(obs)
    rng = np.random.default_rng(1000 + s["seed"])
    x = rng.standard_normal(n)
    cog = np.asarray(s.get("class_of_group", [0] * groups), dtype=np.int64)
    cls = cog[obs["group"]]
    active = np.ones(n, dtype=np.int64)
    if s.get("inactive"):
        active = (np.random.default_rng(3000 + s["seed"]).random(n) >= s["inactive"]).astype(np.int64)
    vecs = {"d": obs["value"].copy(), "x": x}
    out = {}
    out["obs.global_id"] = obs["global_id"].astype(np.int64)
    out["obs.lon"], out["obs.lat"] = obs["x"].copy(), obs["y"].copy()
    out["obs.sigma"] = obs["sigma"].copy()
    out["obs.d"] = vecs["d"]
    out["obs.group"] = obs["group"].astype(np.int64)
    out["obs.class_id"] = cls
    out["obs.active"] = active
    out["vec.x"] = x

    # ---- C1: geometry, GC, correlation blocks ----------------------------------------
    out["c1.xyz"] = points(obs["x"], obs["y"], "sphere")
    c0 = classes[0]["c"]
    r = np.concatenate([[0.0, 0.5 * c0, c0, 1.5 * c0, 2.0 * c0 - 1e-9, 2.0 * c0, 2.5 * c0],
                        np.linspace(0.0, 2.2 * c0, 45)])
    out["c1.gc_r"], out["c1.gc_val"] = r, gaspari_cohn(r, c0)
    for k in range(1, len(classes)):
        ck = classes[k]["c"]
        rk = r * (ck / c0)
        out[f"c1.gc_r_cls{k}"], out[f"c1.gc_val_cls{k}"] = rk, gaspari_cohn(rk, ck)
    ids = obs["global_id"]
    lrng = np.random.default_rng(2000 + s["seed"])
    _block("c1.listA", ids[:min(n, 40)], obs, cls, classes, out)
    _block("c1.listB", ids[lrng.permutation(n)[:min(n, 50)]], obs, cls, classes, out)

    # ---- C3: equal-area bands (bisection owners are in the layouts) ---------------------
    for px, py in BANDS:
        out[f"c3.owner_bands_{px}x{py}"] = SphereDecomposition(px, py).owner(
            obs["x"], obs["y"]).astype(np.int64)

    # ---- layouts: C2, C3 (bisection), C4, C5 ------------------------------------------
    layouts = [(f"relaxed_p{P}", P) for P in RELAXED_PATCHES] + [(f"fixed_p{FIXED_PATCHES}", FIXED_PATCHES)]
    for key, P in layouts:
        _layout(obs, cls, active, classes, P, key, vecs, out)

    # ---- C5: exact global solve and exact matvec (independent of the layout) ---------------
    for v, vec in vecs.items():
        exact, mult = vec.copy(), vec.copy()
        for k, cl in enumerate(classes):
            m = np.flatnonzero((cls == k) & (active == 1))
            if len(m):
                R = build_R(obs["x"][m], obs["y"][m], np.ones(len(m)), cl["c"], cl["alpha"],
                            obs["group"][m], "sphere")
                exact[m] = np.linalg.solve(R, vec[m])
                mult[m] = R @ vec[m]
        out[f"c5.exact_{v}"], out[f"c5.mult_{v}"] = exact, mult

    params = dict(case=name, geometry="sphere", earth_radius_km=EARTH_RADIUS, n_obs=n,
                  n_active=int(active.sum()), pattern=s["pattern"], seed=s["seed"], n_groups=groups,
                  group_layout=s.get("group_layout", "random"), n_classes=len(classes),
                  classes=[dict(c_km=c["c"], support_km=2 * c["c"], alpha=c["alpha"], halo_km=c["h"],
                                halo_over_c=c["h"] / c["c"]) for c in classes],
                  class_of_group=cog.tolist(), correlation="gaspari_cohn_chord",
                  halo_criterion="obs", sigma="1 (R = C)", partition="bisection",
                  relaxed_patches=list(RELAXED_PATCHES), fixed_patches=FIXED_PATCHES,
                  fixed_task_counts=list(FIXED_TASKS), bands=[list(b) for b in BANDS],
                  patch_task_map="block", conventions=CONVENTIONS)
    return params, out


def write_case(name, outdir, allow_dirty=False):
    """Write one case to <outdir>/<name>/ and return that directory."""
    commit, dirty = toy_version()
    if dirty and not allow_dirty:
        raise RuntimeError("the toy checkout has uncommitted changes (tracked files or untracked .py "
                           "files); commit them, or pass allow_dirty / --allow-dirty for a scratch export")
    params, arrays = export_case(name)
    d = Path(outdir) / name
    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for key, a in arrays.items():
        a = np.asarray(a)
        assert a.dtype.kind in "iuf", f"{key}: unsupported dtype {a.dtype}"
        dt = "float64" if a.dtype.kind == "f" else "int64"
        a = np.ascontiguousarray(a, dtype="<f8" if dt == "float64" else "<i8")
        (d / f"{key}.bin").write_bytes(a.tobytes())
        entries.append(dict(name=key, dtype=dt, shape=list(a.shape), file=f"{key}.bin"))
    manifest = dict(format=FORMAT, case=name, toy_commit=commit, toy_dirty=dirty,
                    python=sys.version.split()[0], numpy=np.__version__, scipy=scipy.__version__,
                    params=params, arrays=entries)
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


def case_digest(arrays):
    """Platform-tolerant hash of a case: integers exactly, floats to 9 significant digits."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.asarray(arrays[k])
        h.update(k.encode())
        h.update(a.astype("<i8").tobytes() if a.dtype.kind in "iu"
                 else "".join(f"{v:.8e}," for v in a.ravel()).encode())
    return h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m r_matrix.export", description=__doc__.split("\n")[0])
    ap.add_argument("case", nargs="?", help=f"one of {', '.join(CASES)} or 'all'")
    ap.add_argument("outdir", nargs="?")
    ap.add_argument("--list", action="store_true", help="list the cases")
    ap.add_argument("--allow-dirty", action="store_true", help="export from an uncommitted toy (scratch use)")
    a = ap.parse_args(argv)
    if a.list:
        for k, v in CASES.items():
            print(k, v)
        return 0
    if not a.case or not a.outdir or (a.case != "all" and a.case not in CASES):
        ap.error(f"need <case> in {list(CASES)} or 'all', and <outdir>")
    try:
        for name in (CASES if a.case == "all" else [a.case]):
            print(write_case(name, a.outdir, a.allow_dirty))
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
