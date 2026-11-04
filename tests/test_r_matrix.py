import numpy as np
import pytest

from r_matrix import Config, Decomposition, Domain, MockComm, run
from r_matrix.commmap import build_comm_maps, halo_exchange, halo_exchange_adjoint
from r_matrix.covariance import build_R, gaspari_cohn
from r_matrix.diagnostics import approx_matrix, operator_matrix
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


# ---- symmetrised operator -----------------------------------------------------------

TINY = Config(n_obs=300, h=1.0)   # small enough for operator matrices built column by column


def test_halo_exchange_adjoint_is_adjoint():
    comm, _, owned, _, _ = _setup()
    halos, halo_owner, _ = discover_halos(comm, DEC, owned, 1.0)
    maps = build_comm_maps(comm, owned, halos, halo_owner)
    rng = np.random.default_rng(7)
    x = [rng.standard_normal(len(o)) for o in owned]
    y = [rng.standard_normal(len(hh)) for hh in halos]
    lhs = sum(a @ b for a, b in zip(halo_exchange(comm, maps, x), y))
    rhs = sum(a @ b for a, b in zip(x, halo_exchange_adjoint(comm, maps, y, [len(o) for o in owned])))
    assert lhs == pytest.approx(rhs, rel=1e-13)


def test_assembled_matrix_matches_operator():
    res = run(TINY)
    A_black = operator_matrix(lambda d: res.apply_Rinv(d, tag="diag")[0], res.obs_init)
    np.testing.assert_allclose(approx_matrix(res), A_black, rtol=0, atol=1e-12)


def test_symmetric_operator_is_half_sum_and_symmetric():
    res = run(TINY, symmetric=True)
    S = operator_matrix(lambda d: res.apply_Rinv(d, tag="diag")[0], res.obs_init)
    A = approx_matrix(res)
    assert np.abs(A - A.T).max() > 1e-6                       # the plain operator is not symmetric
    np.testing.assert_allclose(S, 0.5 * (A + A.T), rtol=0, atol=1e-12)
    assert np.abs(S - S.T).max() < 1e-13


@pytest.mark.parametrize("overrides", [dict(px=1, py=1), dict(h=20.0)])
def test_symmetric_equals_global_when_exact(overrides):
    m = run(SMALL, symmetric=True, **overrides).metrics
    assert m["rel_l2"] < 1e-12


@pytest.mark.parametrize("overrides", [dict(pattern="clustered", px=6, py=4),   # empty-core ranks
                                       dict(h=0.0)])                          # no halo at all
def test_symmetric_operator_edge_cases(overrides):
    res = run(Config(n_obs=120, symmetric=True, **overrides))
    if overrides.get("pattern") == "clustered":
        assert any(s.ncore == 0 for s in res.solvers)
    S = operator_matrix(lambda d: res.apply_Rinv(d, tag="diag")[0], res.obs_init)
    A = approx_matrix(res)
    np.testing.assert_allclose(S, 0.5 * (A + A.T), rtol=0, atol=1e-12)


# ---- correlation groups ---------------------------------------------------------------

def test_R_is_block_diagonal_over_groups():
    obs = np.concatenate(make_obs(300, DOM, 1, seed=8, n_groups=3, group_layout="colocated"))
    R = build_R(obs["x"], obs["y"], obs["sigma"], 0.5, 0.8, obs["group"])
    g = obs["group"]
    assert np.all(R[g[:, None] != g[None, :]] == 0.0)
    assert np.linalg.eigvalsh(R).min() > 0                    # colocated points stay SPD


@pytest.mark.parametrize("layout", ["random", "colocated"])
@pytest.mark.parametrize("symmetric", [False, True])
def test_two_groups_equal_two_separate_runs(layout, symmetric):
    # Exact equality relies on the geometric "rect" halo, which does not depend on which
    # groups are present. With "obs" the combined halo is a superset, so results differ
    # at the level of the halo truncation error.
    cfg = Config(n_obs=800, h=1.0, n_groups=2, group_layout=layout, symmetric=symmetric,
                 criterion="rect")
    both = run(cfg)
    z_both = np.concatenate(both.z_orig)
    g_all = np.concatenate([o["group"] for o in both.obs_init])
    for g in (0, 1):
        sub = [o[o["group"] == g] for o in both.obs_init]
        z_g = np.concatenate(run(cfg, obs_init=sub).z_orig)
        np.testing.assert_allclose(z_both[g_all == g], z_g, rtol=0, atol=1e-13)


def test_groups_exact_when_halo_covers_domain():
    m = run(SMALL, n_groups=4, group_layout="colocated", h=20.0, symmetric=True).metrics
    assert m["rel_l2"] < 1e-12


def test_groups_without_core_are_skipped():
    res = run(Config(n_obs=400, n_groups=5, px=8, py=6, h=1.0))
    skipped = 0
    for o, hh, s in zip(res.owned, res.halos, res.solvers):
        halo_only = np.setdiff1d(hh["group"], o["group"])
        if s.ncore == 0 or halo_only.size == 0:
            continue
        skipped += 1
        mask = np.isin(np.concatenate([o["group"], hh["group"]]), halo_only)
        z = s.solve(np.ones(s.n))
        assert np.all(z[mask] == 0.0) and np.all(z[~mask] != 0.0)
    assert skipped > 0                                         # the case really occurs
    assert res.metrics["local_R_bytes_max"] < 8 * res.metrics["local_n_max"] ** 2


def test_groups_symmetric_operator_and_residual():
    res = run(Config(n_obs=240, n_groups=3, group_layout="colocated", h=1.0, symmetric=True,
                     px=6, py=4, pattern="clustered"))
    assert any(s.ncore == 0 for s in res.solvers)
    S = operator_matrix(lambda d: res.apply_Rinv(d, tag="diag")[0], res.obs_init)
    A = approx_matrix(res)
    np.testing.assert_allclose(S, 0.5 * (A + A.T), rtol=0, atol=1e-12)
    m = res.metrics                                            # block-wise matvec is exact
    assert m["residual_distributed"] == pytest.approx(m["residual_global"], rel=1e-10)


def test_group_generator_rejects_bad_input():
    with pytest.raises(ValueError):
        make_obs(10, DOM, 1, n_groups=0)
    with pytest.raises(ValueError):
        make_obs(3, DOM, 1, n_groups=4, group_layout="colocated")
