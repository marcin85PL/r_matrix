"""Round-trip and consistency tests of the golden-data exporter (r_matrix.export)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from r_matrix import export
from r_matrix.covariance import build_R, gaspari_cohn

# Digest of the `tiny` case. A mismatch means the exported numbers changed: inspect the diff,
# and if the change is intended, update this value together with the golden data.
TINY_DIGEST = "efdd3508cfedcdaaa5a46c69285718be72ae8ac702ff7e1449041a341f245b79"


def _load(tmp_path_factory, name):
    d = export.write_case(name, tmp_path_factory.mktemp("golden"), allow_dirty=True)
    return d, *export.read_case(d)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return _load(tmp_path_factory, "tiny")


@pytest.fixture(scope="module")
def classes(tmp_path_factory):
    return _load(tmp_path_factory, "sph_classes")


@pytest.fixture(scope="module")
def inactive(tmp_path_factory):
    return _load(tmp_path_factory, "sph_inactive")


@pytest.fixture(scope="module")
def short_halo(tmp_path_factory):
    return _load(tmp_path_factory, "sph_short_halo")


def test_manifest_describes_every_file(tiny):
    d, man, arrays = tiny
    assert man["format"] == export.FORMAT and man["case"] == "tiny"
    assert len(man["toy_commit"]) in (40, 7) or man["toy_commit"] == "unknown"
    assert man["python"] and man["numpy"] and man["scipy"]
    assert man["params"]["n_obs"] == 24 and man["params"]["halo_criterion"] == "obs"
    assert "bisection_split" in man["params"]["conventions"]
    files = {p.name for p in d.glob("*.bin")}
    assert files == {e["file"] for e in man["arrays"]}
    for e in man["arrays"]:
        n = int(np.prod(e["shape"]))
        assert (d / e["file"]).stat().st_size == 8 * n          # float64 / int64, no header
        assert e["dtype"] in ("float64", "int64")
        assert arrays[e["name"]].shape == tuple(e["shape"])
    json.loads((d / "manifest.json").read_text())


def test_bytes_are_little_endian_and_round_trip(tiny):
    d, man, arrays = tiny
    raw = (d / "obs.global_id.bin").read_bytes()
    ids = arrays["obs.global_id"]
    assert raw == ids.astype("<i8").tobytes()
    assert int.from_bytes(raw[:8], "little") == ids[0]
    assert ids.dtype.kind == "i" and arrays["obs.d"].dtype.kind == "f"


def test_export_is_deterministic(tiny, tmp_path):
    d, _, _ = tiny
    d2 = export.write_case("tiny", tmp_path, allow_dirty=True)
    for p in d.iterdir():
        assert p.read_bytes() == (d2 / p.name).read_bytes(), p.name


def test_pinned_digest_of_tiny(tiny):
    assert export.case_digest(tiny[2]) == TINY_DIGEST


def test_dirty_toy_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(export, "toy_version", lambda: ("a" * 40, True))
    with pytest.raises(RuntimeError, match="uncommitted"):
        export.write_case("tiny", tmp_path)
    monkeypatch.setattr(export, "toy_version", lambda: ("a" * 40, False))
    d = export.write_case("tiny", tmp_path)
    man = json.loads((d / "manifest.json").read_text())
    assert man["toy_commit"] == "a" * 40 and man["toy_dirty"] is False


def test_correlation_blocks_and_gc(tiny):
    _, man, a = tiny
    c, alpha = man["params"]["classes"][0]["c_km"], man["params"]["classes"][0]["alpha"]
    assert np.array_equal(a["c1.gc_val"], gaspari_cohn(a["c1.gc_r"], c))
    assert a["c1.gc_val"][0] == 1.0 and a["c1.gc_val"][list(a["c1.gc_r"]).index(2 * c)] == 0.0
    for L in ("listA", "listB"):
        ids = a[f"c1.{L}.ids"]
        pos = np.searchsorted(np.sort(a["obs.global_id"]), ids)
        idx = np.argsort(a["obs.global_id"])[pos]
        assert np.array_equal(a["obs.global_id"][idx], ids)
        chord = a[f"c1.{L}.chord"]
        xyz = a["c1.xyz"][idx]
        assert np.allclose(chord, np.linalg.norm(xyz[:, None] - xyz[None], axis=2), atol=1e-9)
        corr = a[f"c1.{L}.corr"]
        assert np.allclose(corr, corr.T) and np.allclose(np.diag(corr), 1.0)
        assert np.allclose(corr, alpha * gaspari_cohn(chord, c) + (1 - alpha) * np.eye(len(ids)))


LAYOUTS = [("relaxed_p1", 1), ("relaxed_p2", 2), ("relaxed_p4", 4), ("relaxed_p7", 7), ("fixed_p8", 8)]


def _check_sets(a, key, P, k):
    """Owned/halo ID sets of class k in layout `key` are consistent with the owners."""
    pre = f"layout.{key}.cls{k}."
    oid, ooff = a[pre + "owned_ids"], a[pre + "owned_ids_off"]
    hid, hoff, how = a[pre + "halo_ids"], a[pre + "halo_ids_off"], a[pre + "halo_owner"]
    assert len(ooff) == len(hoff) == P + 1
    owner = a[f"layout.{key}.owner"]
    gid_pos = {int(g): i for i, g in enumerate(a["obs.global_id"])}
    for p in range(P):
        own = oid[ooff[p]:ooff[p + 1]]
        assert np.all(np.diff(own) > 0)                                          # sorted by ID
        assert all(owner[gid_pos[int(g)]] == p for g in own)
        h, ho = hid[hoff[p]:hoff[p + 1]], how[hoff[p]:hoff[p + 1]]
        assert not np.isin(h, own).any() and np.all(ho != p)
        assert np.array_equal(np.lexsort((h, ho)), np.arange(len(h)))            # (owner, id)
        assert np.array_equal(np.bincount(ho, minlength=P), a[pre + "halo_counts"][p])
        assert np.array_equal(owner[[gid_pos[int(g)] for g in h]], ho)           # owners agree
    for nm in ("z_core", "w_core", "adj_back"):
        for v in "dx":
            assert len(a[f"{pre}{nm}_{v}"]) == len(oid)
    assert len(a[pre + "w_halo_d"]) == len(hid)
    return oid


@pytest.mark.parametrize("key,P", LAYOUTS)
def test_layout_sets_are_consistent(tiny, key, P):
    _, _, a = tiny
    oid = _check_sets(a, key, P, 0)
    assert np.array_equal(np.sort(oid), np.sort(a["obs.global_id"]))           # a partition
    if key.startswith("fixed"):
        for t in export.FIXED_TASKS:
            tp = a[f"layout.{key}.task_of_patch_t{t}"]
            assert len(tp) == P and set(tp) == set(range(t))


def test_symmetrised_relation(tiny):
    _, _, a = tiny
    for key, P in LAYOUTS:
        pre = f"layout.{key}.cls0."
        oid = a[pre + "owned_ids"]
        pos = np.argsort(a["obs.global_id"])[np.searchsorted(np.sort(a["obs.global_id"]), oid)]
        for v in "dx":
            zs = 0.5 * (a[f"{pre}z_core_{v}"] + a[f"{pre}w_core_{v}"] + a[f"{pre}adj_back_{v}"])
            assert np.allclose(zs, a[f"layout.{key}.zs_{v}"][pos], atol=1e-14)
            assert np.allclose(a[f"{pre}z_core_{v}"], a[f"layout.{key}.zp_{v}"][pos], atol=1e-14)


def test_operator_outputs(tiny):
    _, man, a = tiny
    cl = man["params"]["classes"][0]
    R = build_R(a["obs.lon"], a["obs.lat"], np.ones(man["params"]["n_obs"]), cl["c_km"], cl["alpha"],
                a["obs.group"], "sphere")
    d, x = a["obs.d"], a["vec.x"]
    assert np.allclose(R @ a["c5.exact_d"], d, atol=1e-10)
    assert np.allclose(R @ a["c5.exact_x"], x, atol=1e-10)
    assert np.array_equal(a["c5.mult_x"], R @ x)
    assert cl["halo_km"] >= 2 * cl["c_km"]                                      # mult exact
    for key, P in LAYOUTS:
        assert np.allclose(a[f"layout.{key}.mult_d"], a["c5.mult_d"], atol=1e-10)
    for v in "dx":
        for nm in ("zs", "zp"):
            assert np.allclose(a[f"layout.relaxed_p1.{nm}_{v}"], a[f"c5.exact_{v}"], atol=1e-10)
    for key in ("relaxed_p2", "relaxed_p4", "relaxed_p7", "fixed_p8"):
        z = a[f"layout.{key}.zs_d"]
        assert 1e-8 < np.linalg.norm(z - a["c5.exact_d"]) / np.linalg.norm(a["c5.exact_d"]) < 1e-2
        assert not np.array_equal(z, a[f"layout.{key}.zp_d"])                   # sym differs from plain


def test_two_classes(classes):
    _, man, a = classes
    p = man["params"]
    assert p["n_classes"] == 2 and p["classes"][0]["c_km"] != p["classes"][1]["c_km"]
    cls = a["obs.class_id"]
    assert set(cls) == {0, 1}
    assert np.array_equal(cls, np.array(p["class_of_group"])[a["obs.group"]])
    assert "c1.gc_val_cls1" in a
    assert np.allclose(a["c1.gc_val_cls1"], gaspari_cohn(a["c1.gc_r_cls1"], p["classes"][1]["c_km"]))
    for k in (0, 1):
        oid = _check_sets(a, "relaxed_p4", 4, k)
        assert set(oid) == set(a["obs.global_id"][cls == k])                    # each class partitioned
    # corr blocks use each datum's class; data of different classes are uncorrelated
    corr = a["c1.listB.corr"]
    idx = np.argsort(a["obs.global_id"])[np.searchsorted(np.sort(a["obs.global_id"]), a["c1.listB.ids"])]
    other = cls[idx][:, None] != cls[idx][None, :]
    assert np.all(corr[other] == 0.0)
    # the exact solve is block-diagonal over classes: it solves each class with its own c, alpha
    for k in (0, 1):
        m = cls == k
        cl = p["classes"][k]
        R = build_R(a["obs.lon"][m], a["obs.lat"][m], np.ones(m.sum()), cl["c_km"], cl["alpha"],
                    a["obs.group"][m], "sphere")
        assert np.allclose(R @ a["c5.exact_d"][m], a["obs.d"][m], atol=1e-9)


def test_inactive_data_pass_through(inactive):
    _, man, a = inactive
    act = a["obs.active"] == 1
    assert 0 < act.sum() < len(act) and man["params"]["n_active"] == act.sum()
    for key, P in LAYOUTS:
        assert np.all(a[f"layout.{key}.owner"][~act] == -1) and np.all(a[f"layout.{key}.owner"][act] >= 0)
        for v in "dx":
            for nm in ("zs", "zp", "mult"):
                assert np.array_equal(a[f"layout.{key}.{nm}_{v}"][~act], a["obs.d" if v == "d" else "vec.x"][~act])
            assert np.array_equal(a[f"c5.exact_{v}"][~act], a["obs.d" if v == "d" else "vec.x"][~act])
        assert set(_check_sets(a, key, P, 0)) == set(a["obs.global_id"][act])


def test_short_halo_mult_is_approximate(short_halo):
    _, man, a = short_halo
    cl = man["params"]["classes"][0]
    assert cl["halo_km"] < 2 * cl["c_km"]
    diff = np.abs(a["layout.relaxed_p4.mult_d"] - a["c5.mult_d"]).max()
    assert diff > 1e-4
    assert np.allclose(a["layout.relaxed_p1.mult_d"], a["c5.mult_d"], atol=1e-10)   # one patch: exact


def test_bands_owners(tiny):
    _, _, a = tiny
    for px, py in export.BANDS:
        o = a[f"c3.owner_bands_{px}x{py}"]
        assert o.min() >= 0 and o.max() < px * py


def test_cli(tmp_path):
    r = subprocess.run([sys.executable, "-m", "r_matrix.export", "tiny", str(tmp_path), "--allow-dirty"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "tiny" / "manifest.json").exists()
    bad = subprocess.run([sys.executable, "-m", "r_matrix.export", "nope", str(tmp_path)],
                         capture_output=True, text=True)
    assert bad.returncode != 0
