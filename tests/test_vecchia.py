"""Tests of the Vecchia prototype (r_matrix/vecchia.py): exactness with complete conditioning sets,
distributed = serial, symmetry and positive definiteness, task-count independence (D2), sampling."""
import numpy as np
import pytest

from r_matrix import vecchia as V
from r_matrix import vecchia_eval as E
from r_matrix.decomposition import BisectionDecomposition
from r_matrix.geometry import points


def _sorted(obs):
    o = obs[np.argsort(obs["global_id"], kind="stable")]
    return o, points(o["x"], o["y"], "sphere"), o["global_id"]


def _gather(built, per_patch):
    """Per-patch vectors (owned order) -> one vector in sorted global-ID order."""
    out = np.zeros(len(built.gids))
    for v, x in zip(built.vowned, per_patch):
        out[np.searchsorted(built.gids, v["global_id"])] = x
    return out


@pytest.fixture(scope="module")
def uniform():
    obs, params, hs, _ = E.golden_data("sph_uniform")
    dec = BisectionDecomposition.from_points(obs["x"], obs["y"], obs["global_id"], 8)
    return obs, params, hs, dec


def test_maxmin_is_a_function_of_the_data():
    obs, *_ = E.golden_data("sph_clustered")
    o, xyz, gid = _sorted(obs)
    perm = np.random.default_rng(1).permutation(len(gid))
    a = gid[V.maxmin_order(xyz, gid)[0]]
    b = gid[perm][V.maxmin_order(xyz[perm], gid[perm])[0]]
    assert np.array_equal(a, b)


@pytest.mark.parametrize("variant", ["point", "block"])
def test_complete_conditioning_is_exact(uniform, variant):
    obs, params, _, dec = uniform
    c, a = params[0]
    o, xyz, gid = _sorted(obs)
    C = E.exact_C(obs, c, a)
    if variant == "point":
        rank = V.rank_of(V.maxmin_order(xyz, gid)[0])
        W = V.point_W(xyz, V.point_neighbours(xyz, rank, len(gid))[0], c, a)
    else:
        owner = dec.owner(o["x"], o["y"])
        seq = np.arange(dec.size)
        W = V.block_W(xyz, gid, owner, seq, V.block_sets(xyz, gid, owner, seq, 1e5), c, a)
    assert V.eig_metrics(V.gen_eigs_W(W, C))["max_dev"] < 1e-12


@pytest.mark.parametrize("order,coarse", [("maxmin", 0), ("maxmin", 25), ("patch", 0)])
def test_distributed_point_equals_serial(uniform, order, coarse):
    obs, params, _, dec = uniform
    c, a = params[0]
    b = E.build(obs, dec, "point", params, m=12, order=order, coarse=coarse, R0=800.0)
    o, xyz, gid = _sorted(obs)
    rk = E.orderings(o, dec.owner(o["x"], o["y"]), b.info["seq"], order)
    rank = np.array([rk[g] for g in gid])
    Ws = V.point_W(xyz, V.point_neighbours(xyz, rank, 12)[0], c, a)
    assert (b.W() != Ws).nnz == 0            # bit for bit: the same neighbours, the same arithmetic


@pytest.mark.parametrize("seq", ["id", "colour"])
def test_distributed_block_equals_serial(uniform, seq):
    obs, params, hs, dec = uniform
    c, a = params[0]
    b = E.build(obs, dec, "block", params, seq=seq, h=hs[0])
    o, xyz, gid = _sorted(obs)
    owner = dec.owner(o["x"], o["y"])
    Ws = V.block_W(xyz, gid, owner, b.info["seq"], V.block_sets(xyz, gid, owner, b.info["seq"], hs[0]),
                   c, a)
    assert abs(b.W() - Ws).max() < 1e-12


@pytest.mark.parametrize("variant", ["point", "block"])
def test_apply_symmetric_and_consistent(uniform, variant):
    obs, params, hs, dec = uniform
    b = E.build(obs, dec, variant, params, m=10, h=hs[0])
    W = b.W()
    rng = np.random.default_rng(2)
    x = [rng.standard_normal(len(v)) for v in b.vowned]
    y = [rng.standard_normal(len(v)) for v in b.vowned]
    Ax = _gather(b, V.apply_inv(b.comm, b.maps, b.factors, x))
    Ay = _gather(b, V.apply_inv(b.comm, b.maps, b.factors, y))
    xg, yg = _gather(b, x), _gather(b, y)
    assert np.allclose(Ax, W.T @ (W @ xg), rtol=0, atol=1e-13)
    assert abs(yg @ Ax - xg @ Ay) < 1e-13 * abs(yg @ Ax)
    # W^T is the adjoint of W, through the halo and adjoint halo exchanges
    Wx = _gather(b, V.apply_W(b.comm, b.maps, b.factors, x))
    Wty = _gather(b, V.apply_Wt(b.comm, b.maps, b.factors, y))
    assert abs(yg @ Wx - xg @ Wty) < 1e-13 * abs(yg @ Wx)


@pytest.mark.parametrize("case,m", [("sph_clustered", 1), ("sph_clustered", 4), ("sph_geo_disc", 3)])
def test_positive_definite_for_any_m(case, m):
    obs, params, _, _ = E.golden_data(case)
    c, a = params[0]
    o, xyz, gid = _sorted(obs)
    rank = V.rank_of(V.maxmin_order(xyz, gid)[0])
    W = V.point_W(xyz, V.point_neighbours(xyz, rank, m)[0], c, a)
    assert V.eig_metrics(V.gen_eigs_W(W, E.exact_C(obs, c, a)))["lam_min"] > 0


@pytest.mark.parametrize("variant", ["point", "block"])
def test_independent_of_task_count(uniform, variant):
    """D2: 8 fixed patches on 1, 2, 4 or 7 tasks give bit-identical R^-1 x and samples."""
    obs, params, hs, dec = uniform
    rng = np.random.default_rng(3)
    ref = None
    for t in (1, 2, 4, 7):
        b = E.build(obs, dec, variant, params, n_tasks=t, m=10, h=hs[0])
        if ref is None:
            x = [rng.standard_normal(len(v)) for v in b.vowned]
        y = _gather(b, V.apply_inv(b.comm, b.maps, b.factors, x))
        s = _gather(b, V.sample(b.comm, b.maps, b.factors, x)[0])
        if ref is None:
            ref = (y, s)
        assert np.array_equal(y, ref[0]) and np.array_equal(s, ref[1])


@pytest.mark.parametrize("variant", ["point", "block"])
def test_sample_solves_W(uniform, variant):
    obs, params, hs, dec = uniform
    b = E.build(obs, dec, variant, params, m=10, seq="colour", h=hs[0])
    z = [np.random.default_rng(4).standard_normal(len(v)) for v in b.vowned]
    x, rounds = V.sample(b.comm, b.maps, b.factors, z)
    assert np.allclose(b.W() @ _gather(b, x), _gather(b, z), rtol=0, atol=1e-12)
    assert rounds >= 1


@pytest.mark.parametrize("case", ["sph_groups", "sph_classes", "sph_poles_dateline", "tiny"])
def test_groups_classes_and_small_cases(case):
    """Groups are independent; groups smaller than m are exact; poles and dateline need nothing
    special (chord distances)."""
    obs, params, hs, _ = E.golden_data(case)
    dec = BisectionDecomposition.from_points(obs["x"], obs["y"], obs["global_id"], 4)
    for variant in ("point", "block"):
        b = E.build(obs, dec, variant, params, m=len(obs), h=1e5, R0=1e5)
        W = b.W()
        o = obs[np.argsort(obs["global_id"], kind="stable")]
        for g, (c, a) in params.items():
            i = np.flatnonzero(o["group"] == g)
            Wg = W[i][:, i]
            assert W[i][:, np.setdiff1d(np.arange(len(o)), i)].nnz == 0
            lam = V.gen_eigs_W(Wg, E.exact_C(o[i], c, a))
            assert np.abs(lam - 1).max() < 1e-10, (case, variant, g)
