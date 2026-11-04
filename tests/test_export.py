"""Round-trip and consistency tests of the golden-data exporter (r_matrix.export)."""
import json
import subprocess
import sys

import numpy as np
import pytest

from r_matrix import export
from r_matrix.covariance import build_R, gaspari_cohn


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    out = tmp_path_factory.mktemp("golden")
    d = export.write_case("tiny", out)
    return d, *export.read_case(d)


def test_manifest_describes_every_file(tiny):
    d, man, arrays = tiny
    assert man["format"] == export.FORMAT and man["case"] == "tiny"
    assert len(man["toy_commit"]) in (7, 40) or man["toy_commit"] == "unknown"
    assert man["params"]["n_obs"] == 24 and man["params"]["halo_criterion"] == "obs"
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
    d2 = export.write_case("tiny", tmp_path)
    for p in d.iterdir():
        assert p.read_bytes() == (d2 / p.name).read_bytes(), p.name


def test_correlation_blocks_and_gc(tiny):
    _, man, a = tiny
    c, alpha = man["params"]["c_km"], man["params"]["alpha"]
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


@pytest.mark.parametrize("key,P", [("relaxed_p1", 1), ("relaxed_p2", 2), ("relaxed_p4", 4),
                                   ("relaxed_p7", 7), ("fixed_p8", 8)])
def test_layout_sets_are_consistent(tiny, key, P):
    _, _, a = tiny
    pre = f"layout.{key}."
    oid, ooff = a[pre + "owned_ids"], a[pre + "owned_ids_off"]
    hid, hoff, how = a[pre + "halo_ids"], a[pre + "halo_ids_off"], a[pre + "halo_owner"]
    assert len(ooff) == len(hoff) == P + 1
    assert np.array_equal(np.sort(oid), np.sort(a["obs.global_id"]))           # a partition
    owner = a[pre + "owner"]
    for p in range(P):
        own = oid[ooff[p]:ooff[p + 1]]
        assert np.all(np.diff(own) > 0)                                          # sorted by ID
        assert np.all(owner[np.isin(a["obs.global_id"], own)] == p)
        h, ho = hid[hoff[p]:hoff[p + 1]], how[hoff[p]:hoff[p + 1]]
        assert not np.isin(h, own).any() and np.all(ho != p)
        order = np.lexsort((h, ho))
        assert np.array_equal(order, np.arange(len(h)))                          # (owner, id)
        assert np.array_equal(np.bincount(ho, minlength=P), a[pre + "halo_counts"][p])
        gid_pos = {int(g): k for k, g in enumerate(a["obs.global_id"])}
        assert np.array_equal(owner[[gid_pos[int(g)] for g in h]], ho)          # owners agree
    assert len(a[pre + "z_core"]) == len(oid) and len(a[pre + "w_halo"]) == len(hid)
    if key.startswith("fixed"):
        for t in export.FIXED_TASKS:
            tp = a[pre + f"task_of_patch_t{t}"]
            assert len(tp) == P and set(tp) == set(range(t))


def test_operator_outputs(tiny):
    _, man, a = tiny
    c, alpha = man["params"]["c_km"], man["params"]["alpha"]
    ones = np.ones(man["params"]["n_obs"])
    R = build_R(a["obs.lon"], a["obs.lat"], ones, c, alpha, a["obs.group"], "sphere")
    d, x = a["obs.d"], a["vec.x"]
    assert np.allclose(R @ a["c5.exact_d"], d, atol=1e-10)
    assert np.allclose(R @ a["c5.exact_x"], x, atol=1e-10)
    assert np.array_equal(a["c5.mult_x"], R @ x)
    # one patch is the exact solve; mult is exact because h >= 2c
    assert man["params"]["halo_km"] >= 2 * c
    assert np.allclose(a["layout.relaxed_p1.zs_d"], a["c5.exact_d"], atol=1e-10)
    assert np.allclose(a["layout.relaxed_p1.zp_d"], a["c5.exact_d"], atol=1e-10)
    for key in ("relaxed_p2", "relaxed_p4", "relaxed_p7", "fixed_p8"):
        z = a[f"layout.{key}.zs_d"]
        assert 1e-8 < np.linalg.norm(z - a["c5.exact_d"]) / np.linalg.norm(a["c5.exact_d"]) < 1e-2
        assert not np.array_equal(z, a[f"layout.{key}.zp_d"])                   # sym differs from plain


def test_bands_owners(tiny):
    _, _, a = tiny
    for px, py in export.BANDS:
        o = a[f"c3.owner_bands_{px}x{py}"]
        assert o.min() >= 0 and o.max() < px * py


def test_cli(tmp_path):
    r = subprocess.run([sys.executable, "-m", "r_matrix.export", "tiny", str(tmp_path)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "tiny" / "manifest.json").exists()
    bad = subprocess.run([sys.executable, "-m", "r_matrix.export", "nope", str(tmp_path)],
                         capture_output=True, text=True)
    assert bad.returncode != 0
