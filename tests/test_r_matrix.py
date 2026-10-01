import numpy as np
import pytest

from r_matrix import Config, Decomposition, Domain, MockComm, run
from r_matrix.commmap import build_comm_maps, halo_exchange
from r_matrix.covariance import build_R, gaspari_cohn
from r_matrix.generate import make_obs
from r_matrix.halo import discover_halos
from r_matrix.redistribute import check_redistribution, redistribute

DOM = Domain(0.0, 12.0, 0.0, 9.0)
DEC = Decomposition(DOM, 4, 3)
SMALL = Config(n_obs=600)


def _setup(n=800, seed=1, pattern="uniform", **kw):
    obs = make_obs(n, DOM, DEC.size, seed=seed, pattern=pattern, decomposition=DEC, **kw)
    comm = MockComm(DEC.size)
    owned, rejected, plan = redistribute(comm, DEC, obs)
    return comm, obs, owned, rejected, plan


# ---- ownership --------------------------------------------------------------------

def test_owner_formula_and_clamp():
    x = np.array([0.0, 2.999, 3.0, 12.0, 11.99, 0.0, 12.0])
    y = np.array([0.0, 0.0, 0.0, 0.0, 8.99, 9.0, 9.0])
    assert DEC.owner(x, y).tolist() == [0, 0, 1, 3, 11, 8, 11]


def test_ranks_intersecting_and_rect_distance():
    assert sorted(DEC.ranks_intersecting(2.5, 3.5, 2.5, 3.5).tolist()) == [0, 1, 4, 5]
    d = DEC.rect_distance(5, np.array([4.0, 2.0, 2.0]), np.array([4.0, 4.0, 2.0]))
    np.testing.assert_allclose(d, [0.0, 1.0, np.sqrt(2.0)])


def test_gaspari_cohn_properties():
    r = np.linspace(0, 3, 301)
    g = gaspari_cohn(r, 1.0)
    assert g[0] == pytest.approx(1.0)
    assert np.all(g[r >= 2.0] == 0.0)
    assert np.all(np.diff(g) <= 1e-12)
    assert np.ptp(gaspari_cohn(np.array([1.0 - 1e-9, 1.0 + 1e-9]), 1.0)) < 1e-7  # continuous at z=1


# ---- redistribution ---------------------------------------------------------------

@pytest.mark.parametrize("pattern", ["uniform", "clustered", "coincident", "boundary", "geostationary"])
def test_redistribution_invariants(pattern):
    comm, obs, owned, rejected, _ = _setup(pattern=pattern)
    check_redistribution(obs, owned, rejected, DEC)
    for o in owned:
        assert np.all(np.diff(o["global_id"]) > 0)


def test_rejects_invalid_and_round_trip():
    obs = make_obs(500, DOM, DEC.size, seed=3)
    obs[0]["x"][0] = np.nan
    obs[1]["y"][1] = 99.0
    comm = MockComm(DEC.size)
    owned, rejected, plan = redistribute(comm, DEC, obs)
    assert sum(len(r) for r in rejected) == 2
    check_redistribution(obs, owned, rejected, DEC)
    # forward then reverse is the identity on valid slots, NaN on rejected ones
    vals = [o["value"] for o in obs]
    back = plan.reverse(comm, plan.forward(comm, vals))
    assert np.isnan(back[0][0]) and np.isnan(back[1][1])
    for v, b in zip(vals, back):
        ok = np.isfinite(b)
        np.testing.assert_array_equal(v[ok], b[ok])
    # forward delivers values in owned (global_id) order
    for o, f in zip(owned, plan.forward(comm, vals)):
        np.testing.assert_array_equal(o["value"], f)


def test_empty_and_skewed_initial_ranks():
    comm, obs, owned, rejected, _ = _setup(empty_ranks=(0, 5, 7), skew=0.3)
    assert len(obs[0]) == len(obs[5]) == len(obs[7]) == 0
    check_redistribution(obs, owned, rejected, DEC)


def test_duplicate_ids_detected():
    obs = make_obs(300, DOM, DEC.size, seed=4)
    obs[2]["global_id"][0] = obs[9]["global_id"][0]
    obs[2]["x"][0], obs[2]["y"][0] = obs[9]["x"][0], obs[9]["y"][0]  # same spatial owner
    with pytest.raises(ValueError, match="duplicate"):
        redistribute(MockComm(DEC.size), DEC, obs)


# ---- halo + map -------------------------------------------------------------------

def _brute_halo_ids(owned, r, h, criterion):
    allobs = np.concatenate([o for s, o in enumerate(owned) if s != r])
    if criterion == "rect":
        keep = DEC.rect_distance(r, allobs["x"], allobs["y"]) <= h
    else:
        core = owned[r]
        d = np.hypot(allobs["x"][:, None] - core["x"][None], allobs["y"][:, None] - core["y"][None])
        keep = d.min(axis=1) <= h if len(core) else np.zeros(len(allobs), bool)
    return np.sort(allobs["global_id"][keep])


@pytest.mark.parametrize("criterion", ["rect", "obs"])
def test_halo_matches_global_brute_force(criterion):
    comm, _, owned, _, _ = _setup()
    halos, _, _ = discover_halos(comm, DEC, owned, 1.3, criterion)
    for r in range(DEC.size):
        np.testing.assert_array_equal(np.sort(halos[r]["global_id"]),
                                      _brute_halo_ids(owned, r, 1.3, criterion))


def test_geometric_and_kdtree_halos_agree():
    comm, _, owned, _, _ = _setup(pattern="clustered")
    a, ao, _ = discover_halos(comm, DEC, owned, 0.9, "obs", "brute")
    b, bo, _ = discover_halos(comm, DEC, owned, 0.9, "obs", "kdtree")
    for r in range(DEC.size):
        np.testing.assert_array_equal(a[r]["global_id"], b[r]["global_id"])
        np.testing.assert_array_equal(ao[r], bo[r])


def test_comm_map_reproduces_owner_values():
    comm, _, owned, _, _ = _setup()
    halos, halo_owner, _ = discover_halos(comm, DEC, owned, 1.0)
    maps = build_comm_maps(comm, owned, halos, halo_owner)
    got = halo_exchange(comm, maps, [o["value"] for o in owned])
    for hh, g in zip(halos, got):
        np.testing.assert_array_equal(hh["value"], g)
    # zero-count peers are omitted from the recurring exchange
    assert all(len(idx) > 0 for m in maps for idx in m.send.values())


# ---- numerics ---------------------------------------------------------------------

def test_single_rank_equals_global():
    m = run(SMALL, px=1, py=1).metrics
    assert m["rel_l2"] < 1e-12


def test_halo_covering_domain_equals_global():
    m = run(SMALL, h=20.0).metrics
    assert m["rel_l2"] < 1e-12 and m["residual_global"] < 1e-12


def test_error_decreases_with_halo():
    errs = [run(SMALL, h=h).metrics["rel_l2"] for h in (0.5, 1.0, 2.0, 3.0)]
    assert all(b < a for a, b in zip(errs, errs[1:]))


def test_distributed_matvec_is_exact_when_halo_covers_support():
    m = run(SMALL, h=1.0).metrics   # support 2c = 1.0
    assert m["matvec_exact"]
    assert m["residual_distributed"] == pytest.approx(m["residual_global"], rel=1e-10)


@pytest.mark.parametrize("pattern", ["clustered", "coincident", "boundary", "geostationary"])
def test_robust_patterns(pattern):
    m = run(SMALL, pattern=pattern, h=2.0).metrics
    assert np.isfinite(m["rel_l2"]) and m["rel_l2"] < 0.05


def test_R_is_spd_with_coincident_points():
    obs = np.concatenate(make_obs(400, DOM, 1, seed=5, pattern="coincident"))
    R = build_R(obs["x"], obs["y"], obs["sigma"], 0.5, 0.8)
    assert np.linalg.eigvalsh(R).min() > 0


def test_geostationary_grid_is_near_uniform():
    from r_matrix.generate import geostationary_shape
    obs = np.concatenate(make_obs(3000, DOM, DEC.size, seed=6, pattern="geostationary"))
    nx, ny = geostationary_shape(3000, DOM)
    assert len(obs) == nx * ny
    # every pixel cell holds exactly one observation (the jitter stays inside the pixel)
    i = np.floor(obs["x"] / (DOM.lx / nx)).astype(int)
    j = np.floor(obs["y"] / (DOM.ly / ny)).astype(int)
    assert np.unique(i + nx * j).size == nx * ny


def test_apply_Rinv_reuses_frozen_setup():
    res = run(SMALL, h=1.5)
    d = [o["value"] for o in res.obs_init]
    z, _ = res.apply_Rinv(d, tag="test")
    for a, b in zip(z, res.z_orig):
        np.testing.assert_allclose(a, b, rtol=0, atol=0)
    # exact path agrees with the stored reference
    for a, b in zip(res.apply_Rinv_exact(d), res.z_ref_orig):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    assert res.comm.stats.phases["test:halo"].bytes == res.comm.stats.phases["recurring:halo"].bytes
