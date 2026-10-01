"""Seeded mock observations with random initial MPI ownership (deck slide 2).

This imitates the situation after observation processing in an NWP code. Each MPI
task holds a batch of observations whose locations have nothing to do with the
task number (think of reports read in file order). Random placement is the test
condition that stresses the redistribution. It is not part of the spatial model.
Everything is drawn from one seeded generator, so any failure can be reproduced
exactly.

Observation layouts (``pattern``)
---------------------------------
uniform     i.i.d. uniform over the domain
clustered   80 % in a few tight Gaussian blobs, 20 % uniform background; tests load
            imbalance and very large halos
coincident  20 % of the points duplicate another point's location, with distinct IDs
            (e.g. several reports from one station); tests conditioning
boundary    20 % snapped exactly onto internal cell edges and onto xmax / ymax; tests
            the [low, high) ownership rule and the clamp
geostationary
            mimics a geostationary imager: a regular pixel grid covering the whole
            domain, each pixel jittered by up to +-15 % of the pixel spacing
            (navigation noise). The coverage is near-uniform, with no gaps, so fields
            can be shown as contour maps. The grid is nx x ny with nx * ny close to
            n and square pixels, so the actual count can differ slightly from n.
"""
import numpy as np
from scipy.linalg import cholesky

from .covariance import build_R
from .records import empty

PATTERNS = ("uniform", "clustered", "coincident", "boundary", "geostationary")


def geostationary_shape(n, domain):
    """Pixel grid (nx, ny) with about n pixels and square pixels over the domain."""
    nx = max(int(round(np.sqrt(n * domain.lx / domain.ly))), 1)
    ny = max(int(round(n / nx)), 1)
    return nx, ny


def _positions(n, domain, pattern, rng, decomposition=None, frac=0.2, n_clusters=6, grid=None):
    """Draw n positions for the given pattern (see the module docstring).

    grid : (nx, ny) pixel grid for "geostationary", with nx * ny == n.
    """
    x = rng.uniform(domain.xmin, domain.xmax, n)
    y = rng.uniform(domain.ymin, domain.ymax, n)
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; expected one of {PATTERNS}")
    if pattern == "clustered":
        m = int(round((1 - frac) * n))
        cx = rng.uniform(domain.xmin, domain.xmax, n_clusters)
        cy = rng.uniform(domain.ymin, domain.ymax, n_clusters)
        width = 0.03 * min(domain.lx, domain.ly)
        k = rng.integers(0, n_clusters, m)
        x[:m] = np.clip(cx[k] + width * rng.standard_normal(m), domain.xmin, domain.xmax)
        y[:m] = np.clip(cy[k] + width * rng.standard_normal(m), domain.ymin, domain.ymax)
    elif pattern == "coincident":
        m = int(round(frac * n))
        src = rng.integers(m, n, m)
        x[:m], y[:m] = x[src], y[src]
    elif pattern == "boundary":
        if decomposition is None:
            raise ValueError("pattern 'boundary' needs the decomposition")
        m = int(round(frac * n))
        # Internal edges plus the upper domain edges (index px / py is xmax / ymax).
        xs = domain.xmin + decomposition.dx * np.arange(1, decomposition.px + 1)
        ys = domain.ymin + decomposition.dy * np.arange(1, decomposition.py + 1)
        half = m // 2
        x[:half] = rng.choice(xs, half)
        y[half:m] = rng.choice(ys, m - half)
    elif pattern == "geostationary":
        nx, ny = grid
        sx, sy = domain.lx / nx, domain.ly / ny
        gx, gy = np.meshgrid(domain.xmin + sx * (np.arange(nx) + 0.5),
                             domain.ymin + sy * (np.arange(ny) + 0.5))
        x = np.clip(gx.ravel() + 0.15 * sx * rng.uniform(-1, 1, n), domain.xmin, domain.xmax)
        y = np.clip(gy.ravel() + 0.15 * sy * rng.uniform(-1, 1, n), domain.ymin, domain.ymax)
    return x, y


def make_obs(n, domain, nranks, seed=0, pattern="uniform", decomposition=None,
             departures="white", c=None, alpha=None, sigma_range=(0.8, 1.2),
             empty_ranks=(), skew=None):
    """Generate n observations scattered at random over nranks initial MPI tasks.

    Parameters
    ----------
    n : number of observations (approximate for pattern="geostationary")
    domain : decomposition.Domain
    nranks : number of initial tasks (the same tasks later own the subdomains)
    seed : seed for numpy.random.default_rng; fixes positions, values and ranks
    pattern : observation layout, one of PATTERNS
    decomposition : needed only for pattern="boundary"
    departures : how the departure vector d (field "value") is drawn
        "white"      -> d_i ~ N(0, sigma_i^2), independent
        "correlated" -> d ~ N(0, R), the realistic choice if R is the true error
                        covariance. Needs c and alpha, and uses a dense global
                        Cholesky, so keep n moderate.
    sigma_range : sigma_i ~ U(sigma_range)
    empty_ranks : initial ranks that get no observations (edge-case testing)
    skew : if given, the rank shares are drawn from Dirichlet(skew); a small skew
        gives very uneven initial counts. None means uniformly random ranks.

    Returns
    -------
    list of nranks OBS_DTYPE arrays. Global IDs are unique but shuffled and
    non-contiguous (1000 + 7k), so no code can rely on id == index. orig_rank and
    orig_index record each observation's initial position.
    """
    rng = np.random.default_rng(seed)
    grid = geostationary_shape(n, domain) if pattern == "geostationary" else None
    if grid is not None:
        n = grid[0] * grid[1]
    obs = empty(n)
    obs["global_id"] = 1000 + 7 * rng.permutation(n).astype(np.int64)
    obs["x"], obs["y"] = _positions(n, domain, pattern, rng, decomposition, grid=grid)
    obs["sigma"] = rng.uniform(*sigma_range, n)
    if departures == "white":
        obs["value"] = obs["sigma"] * rng.standard_normal(n)
    elif departures == "correlated":
        R = build_R(obs["x"], obs["y"], obs["sigma"], c, alpha)
        obs["value"] = cholesky(R, lower=True) @ rng.standard_normal(n)
    else:
        raise ValueError(f"unknown departures {departures!r}")

    active = np.setdiff1d(np.arange(nranks), np.asarray(empty_ranks, dtype=int))
    p = None if skew is None else rng.dirichlet(np.full(active.size, skew))
    ranks = rng.choice(active, n, p=p)
    out = []
    for r in range(nranks):
        o = obs[ranks == r].copy()
        o["orig_rank"] = r
        o["orig_index"] = np.arange(len(o))
        out.append(o)
    return out
