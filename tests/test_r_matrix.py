import numpy as np
import pytest
from scipy.spatial import cKDTree

from r_matrix import Config, Decomposition, Domain, MockComm, SphereDecomposition, run
from r_matrix.commmap import build_comm_maps, halo_exchange, halo_exchange_adjoint
from r_matrix.covariance import build_R, gaspari_cohn
from r_matrix.diagnostics import approx_matrix, operator_matrix
from r_matrix.generate import make_obs
from r_matrix.geometry import EARTH_RADIUS, points
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


# ---- sphere ---------------------------------------------------------------------------

SDEC = SphereDecomposition(4, 3)
SPH = Config(geometry="sphere", n_obs=600, c=1000.0, h=2000.0)   # km; about 1 obs per l^2


def _sphere_setup(n=800, seed=1, pattern="poles_dateline"):
    obs = make_obs(n, None, SDEC.size, seed=seed, pattern=pattern, geometry="sphere")
    comm = MockComm(SDEC.size)
    owned, rejected, plan = redistribute(comm, SDEC, obs)
    return comm, obs, owned, rejected, plan


def test_sphere_owner_wraps_dateline_and_handles_poles():
    lon = np.array([np.pi, -np.pi, 2 * np.pi, 3 * np.pi / 2, -np.pi / 2, 0.3, 0.3])
    lat = np.array([0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2, -np.pi / 2])
    own = SDEC.owner(lon, lat)
    assert own[0] == own[1] == 4                      # 180E = 180W, sector 0, middle band
    assert own[2] == SDEC.owner(0.0, 0.0)              # 2 pi = 0
    assert own[3] == own[4]                            # 270E = 90W
    assert own[5] // SDEC.px == SDEC.py - 1 and own[6] // SDEC.px == 0


def test_sphere_cells_have_equal_area():
    rng = np.random.default_rng(3)
    z, lon = rng.uniform(-1, 1, 200000), rng.uniform(-np.pi, np.pi, 200000)
    counts = np.bincount(SDEC.owner(lon, np.arcsin(z)), minlength=SDEC.size)
    assert counts.std() / counts.mean() < 0.02


@pytest.mark.parametrize("px, py, rank", [(4, 3, 0), (4, 3, 5), (4, 3, 10),   # S wedge, middle, N wedge
                                          (1, 3, 1), (2, 1, 0), (2, 2, 3), (8, 6, 47)])
def test_sphere_rect_distance_matches_dense_sampling(px, py, rank):
    dec = SphereDecomposition(px, py)
    rng = np.random.default_rng(rank + 10 * px)
    lon0, lon1, lat0, lat1 = dec.cell_rect(rank)
    m = 600
    g_lon, g_lat = np.meshgrid(np.linspace(lon0, lon1, m), np.linspace(lat0, lat1, m))
    cell = points(g_lon.ravel(), g_lat.ravel(), "sphere")
    lon = rng.uniform(-np.pi, 3 * np.pi, 300)                       # includes unwrapped lon
    lat = np.arcsin(rng.uniform(-1, 1, 300))
    lat[:10] = np.pi / 2                                            # some exactly at the poles
    lat[10:20] = -np.pi / 2
    edges = -np.pi + 2 * np.pi * np.arange(px + 1) / px
    lon[20:40] = edges[rng.integers(0, px + 1, 20)] + 2 * np.pi * rng.integers(-1, 2, 20)
    p = points(lon, lat, "sphere")
    brute = cKDTree(cell).query(p)[0]                               # nearest sample point
    exact = dec.rect_distance(rank, lon, lat)
    # Every cell point is within half a grid diagonal of a sample point.
    tol = EARTH_RADIUS * 0.5 * np.hypot((lon1 - lon0) / (m - 1), (lat1 - lat0) / (m - 1)) + 1e-6
    assert np.all(exact <= brute + 1e-6)                            # never larger than any cell point
    assert np.all(brute - exact <= tol)
    inside = dec.owner(lon, lat) == rank
    assert np.all(exact[inside] == 0.0)


def test_sphere_rect_distance_accepts_scalars():
    assert SDEC.rect_distance(5, 0.0, 0.0) == pytest.approx(SDEC.rect_distance(5, np.array([0.0]), np.array([0.0]))[0])


def test_sphere_rect_halo_contains_obs_halo():
    comm, _, owned, _, _ = _sphere_setup(pattern="clustered")
    rect, _, _ = discover_halos(comm, SDEC, owned, 1200.0, "rect")
    obs, _, _ = discover_halos(comm, SDEC, owned, 1200.0, "obs")
    for a, b in zip(rect, obs):
        assert np.isin(b["global_id"], a["global_id"]).all()


def test_sphere_small_n_and_bad_geometry():
    assert sum(len(o) for o in make_obs(10, None, 2, pattern="poles_dateline", geometry="sphere")) == 10
    with pytest.raises(ValueError, match="geometry"):
        run(SPH, geometry="spher")


@pytest.mark.parametrize("pattern", ["uniform", "clustered", "poles_dateline"])
def test_sphere_redistribution_invariants(pattern):
    comm, obs, owned, rejected, _ = _sphere_setup(pattern=pattern)
    assert sum(len(r) for r in rejected) == 0          # lon outside [-pi, pi) is valid
    check_redistribution(obs, owned, rejected, SDEC)


@pytest.mark.parametrize("criterion", ["rect", "obs"])
def test_sphere_halo_matches_global_brute_force(criterion):
    comm, _, owned, _, _ = _sphere_setup()
    h = 1500.0
    halos, _, _ = discover_halos(comm, SDEC, owned, h, criterion)
    for r in range(SDEC.size):
        others = np.concatenate([o for s, o in enumerate(owned) if s != r])
        if criterion == "rect":
            keep = SDEC.rect_distance(r, others["x"], others["y"]) <= h
        else:
            a = points(others["x"], others["y"], "sphere")
            b = points(owned[r]["x"], owned[r]["y"], "sphere")
            keep = np.linalg.norm(a[:, None] - b[None], axis=2).min(axis=1) <= h
        np.testing.assert_array_equal(np.sort(halos[r]["global_id"]), np.sort(others["global_id"][keep]))


def test_sphere_kdtree_and_brute_halos_agree():
    comm, _, owned, _, _ = _sphere_setup(pattern="clustered")
    a, _, _ = discover_halos(comm, SDEC, owned, 900.0, "obs", "brute")
    b, _, _ = discover_halos(comm, SDEC, owned, 900.0, "obs", "kdtree")
    for r in range(SDEC.size):
        np.testing.assert_array_equal(a[r]["global_id"], b[r]["global_id"])


@pytest.mark.parametrize("overrides", [dict(px=1, py=1), dict(h=13000.0), dict(h=13000.0, symmetric=True)])
def test_sphere_exact_cases(overrides):        # 13000 km > the sphere's diameter
    assert run(SPH, pattern="poles_dateline", **overrides).metrics["rel_l2"] < 1e-12


@pytest.mark.parametrize("pattern", ["uniform", "poles_dateline"])
def test_sphere_matvec_exact_when_halo_covers_support(pattern):
    m = run(SPH, pattern=pattern).metrics            # h = 2c
    assert m["matvec_exact"]
    assert m["residual_distributed"] == pytest.approx(m["residual_global"], rel=1e-10)


def test_sphere_error_decreases_with_halo():
    errs = [run(SPH, h=h, symmetric=True).metrics["rel_l2"] for h in (500.0, 1000.0, 2000.0, 3000.0)]
    assert all(b < a for a, b in zip(errs, errs[1:]))


def test_sphere_exact_solution_invariant_to_longitude_shift():
    res = run(SPH, pattern="poles_dateline", reference=True)
    rot = [o.copy() for o in res.obs_init]
    for o in rot:
        o["x"] += 1.234                                # shift lon (rotation about the polar axis);
                                                       # also makes some lon > pi, i.e. unwrapped
    res2 = run(SPH, obs_init=rot)
    for a, b in zip(res.z_ref_orig, res2.z_ref_orig):
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)


def test_sphere_groups_equal_separate_runs():
    cfg = Config(geometry="sphere", n_obs=600, c=1000.0, h=2000.0, n_groups=2,
                               symmetric=True, pattern="poles_dateline")
    both = run(cfg)
    z_both = np.concatenate(both.z_orig)
    g_all = np.concatenate([o["group"] for o in both.obs_init])
    for g in (0, 1):
        sub = [o[o["group"] == g] for o in both.obs_init]
        np.testing.assert_allclose(z_both[g_all == g], np.concatenate(run(cfg, obs_init=sub).z_orig),
                                   rtol=0, atol=1e-13)


# ---- R-patches separate from MPI tasks (reproducible mode) ---------------------------

def test_patch_task_map():
    from r_matrix.pipeline import patch_task_map
    t, home = patch_task_map(12, 5, "block")
    assert t.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4] and home.tolist() == [0, 3, 5, 8, 10]
    t, home = patch_task_map(12, 5, "round_robin")
    assert t.tolist() == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1] and home.tolist() == [0, 1, 2, 3, 4]
    with pytest.raises(ValueError):
        patch_task_map(12, 13)


def _z_by_gid(res):
    g = np.concatenate([o["global_id"] for o in res.obs_init])
    z = np.concatenate(res.z_orig)
    order = np.argsort(g)
    return g[order], z[order]


@pytest.mark.parametrize("base", [
    Config(n_obs=800, h=1.5, symmetric=True),
    Config(n_obs=800, h=1.5, symmetric=True, n_groups=3, group_layout="colocated", pattern="clustered"),
    Config(n_obs=800, geometry="sphere", c=1000.0, h=3000.0, px=6, py=4, symmetric=True,
           pattern="poles_dateline"),
    Config(n_obs=800, h=1.3, criterion="obs", halo_method="kdtree", skew=0.5),
], ids=["plane", "groups", "sphere", "plain-obs-kdtree-skew"])
def test_results_bit_identical_for_any_task_count(base):
    # Bitwise equality also relies on BLAS/LAPACK giving identical results for identical
    # matrices, which holds in this single process.
    r0 = run(base, n_tasks=1)
    g0, z0 = _z_by_gid(r0)
    cases = [(4, "block"), (12, "block"), (5, "round_robin"), (None, "block")]
    if base.geometry == "sphere":
        cases.append((24, "block"))
    for T, how in cases:
        r = run(base, n_tasks=T, patch_map=how,
                empty_init_ranks=(1,) if T not in (1, None) and base.skew else ())
        g, z = _z_by_gid(r)
        assert np.array_equal(g, g0) and np.array_equal(z, z0)        # bitwise, not approx
        assert r.metrics["rel_l2"] == r0.metrics["rel_l2"]            # reduced in gid order


def test_traffic_counted_between_tasks_only():
    one = run(SMALL, n_tasks=1).metrics
    assert one["bytes_apply"] == 0 and one["messages_apply"] == 0 and one["bytes_setup"] == 0
    four, all12 = run(SMALL, n_tasks=4).metrics, run(SMALL).metrics
    assert 0 < four["bytes_apply"] < all12["bytes_apply"]
    assert four["owned_task"]["mean"] == pytest.approx(SMALL.n_obs / 4)


def test_patches_apply_Rinv_reuses_setup_and_checks_input():
    res = run(SMALL, n_tasks=5, patch_map="round_robin")
    z, _ = res.apply_Rinv([o["value"] for o in res.obs_init], tag="test")
    for a, b in zip(z, res.z_orig):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match="task arrays"):
        run(SMALL, n_tasks=5, obs_init=res.obs_init[:4])
    with pytest.raises(ValueError, match="one array per task"):
        res.apply_Rinv([o["value"] for o in res.obs_init] + [np.zeros(0)])
    for a, b in zip(res.apply_Rinv_exact([o["value"] for o in res.obs_init]), res.z_ref_orig):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_commstats_aggregates_to_task_pairs():
    from r_matrix.comm import CommStats
    pb = np.array([[0, 5, 7, 0],          # patches 0,1 on task 0; patches 2,3 on task 1
                   [3, 0, 0, 2],
                   [0, 0, 0, 4],
                   [1, 0, 6, 0]])
    st = CommStats(task_of=np.array([0, 0, 1, 1]))
    st.record("x", pb)
    ph = st.phases["x"]
    assert ph.bytes == 7 + 2 + 1             # only cross-task entries
    assert ph.messages == 2 and ph.max_peers == 1


def test_sweep_flattens_task_metrics():
    from r_matrix import sweep
    rows = sweep("h", [1.0], Config(n_obs=300, n_tasks=4))
    assert rows[0]["owned_task_mean"] == pytest.approx(75.0) and rows[0]["owned_mean"] == pytest.approx(25.0)
