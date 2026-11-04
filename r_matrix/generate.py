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
from scipy.spatial import cKDTree

from .covariance import build_R
from .records import empty

PATTERNS = ("uniform", "clustered", "coincident", "boundary", "geostationary")
PATTERNS_SPHERE = ("uniform", "clustered", "poles_dateline", "geo_disc")  # geo_disc: see _geo_disc


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


def _positions_sphere(n, pattern, rng, frac=0.2, n_clusters=6):
    """Draw n (lon, lat) positions in radians on the sphere.

    uniform         uniform in area (lon uniform, sin(lat) uniform)
    clustered       80 % in n_clusters tight blobs (width about 0.03 rad), 20 % uniform
    geo_disc        a geostationary imager after thinning, modelled on SEVIRI clear-sky
                    radiances in an IFS CCMA: a near-uniform lattice (Fibonacci, jittered)
                    over the disc within 66 deg of the sub-satellite point (lon 0), from
                    which cloudy pixels are removed by thresholding a smooth random field
                    (cloud systems of about 500 km). The lattice spacing is fixed and
                    calibrated on the 2022060100 CCMA of experiment jdfp: nearest-
                    neighbour distances 43 / 56 / 67 km (10th / 50th / 90th percentile)
                    for 11000 pixels; the CCMA has 49 / 55 / 79 km, and the same median
                    (54-55 km) for every channel. n is the number of clear pixels (at
                    most n) and sets the clear fraction (0.35 for 11000).
    poles_dateline  stresses the special places: of the 20 % special points, a quarter
                    lie exactly on lon = +pi, a quarter exactly on lon = -pi, a quarter
                    within 0.03 rad of a pole (some exactly on it), and a quarter have
                    lon shifted by +2 pi, i.e. stored in [pi, 3 pi) as an ODB using
                    [0, 2 pi) would
    """
    if pattern not in PATTERNS_SPHERE:
        raise ValueError(f"unknown sphere pattern {pattern!r}; expected one of {PATTERNS_SPHERE}")
    lon = rng.uniform(-np.pi, np.pi, n)
    lat = np.arcsin(rng.uniform(-1.0, 1.0, n))
    if pattern == "clustered":
        m = int(round((1 - frac) * n))
        cen = rng.standard_normal((n_clusters, 3))
        cen /= np.linalg.norm(cen, axis=1, keepdims=True)
        v = cen[rng.integers(0, n_clusters, m)] + 0.03 * rng.standard_normal((m, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        lon[:m], lat[:m] = np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2], -1, 1))
    elif pattern == "poles_dateline":
        q = int(round(frac * n)) // 4
        lon[:q] = np.pi
        lon[q:2 * q] = -np.pi
        if q > 0:                      # n < 20 leaves no special points
            sign = rng.choice([-1.0, 1.0], q)
            lat[2 * q:3 * q] = sign * (np.pi / 2 - rng.uniform(0.0, 0.03, q))
            k = max(q // 4, 1)
            lat[2 * q:2 * q + k] = sign[:k] * np.pi / 2
            lon[3 * q:4 * q] += 2.0 * np.pi
    return lon, lat


SEVIRI_SPACING_N = 11000 / 0.35   # lattice points in the disc, see _geo_disc


def _geo_disc(n, rng, sub_lon=0.0, max_angle=np.radians(66.0), cloud_km=500.0):
    """Clear pixels of one geostationary image (see _positions_sphere, pattern geo_disc).

    The lattice is fixed: about SEVIRI_SPACING_N points in the disc, which after jitter
    gives nearest-neighbour distances with a median of about 56 km. In the CCMA every
    SEVIRI channel has a median spacing of 54-55 km; channels with fewer active pixels
    are not sparser, they have more of the disc masked. So n sets the clear fraction,
    n / (lattice points in the disc), not the spacing.
    """
    cap = (1.0 - np.cos(max_angle)) / 2.0                 # disc area / sphere area
    n_lattice = int(round(SEVIRI_SPACING_N / cap))        # lattice points over the whole sphere
    k = np.arange(n_lattice) + 0.5
    z = 1.0 - 2.0 * k / n_lattice
    phi = np.pi * (1.0 + np.sqrt(5.0)) * k                # golden-angle spiral
    v = np.column_stack([np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z])
    centre = np.array([np.cos(sub_lon), np.sin(sub_lon), 0.0])
    v = v[v @ centre >= np.cos(max_angle)]
    spacing = np.sqrt(4.0 * np.pi / n_lattice)           # radians, about the lattice step
    v = v + 0.15 * spacing * rng.standard_normal(v.shape)  # navigation / box-centre jitter
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    if n <= 0:
        return np.zeros(0), np.zeros(0)
    # Cloud field: a sum of Gaussian bumps of width w = cloud_km, about 4 per w^2 of
    # area, with centres uniform over the sphere. Each bump is evaluated only within 5 w
    # of its centre (k-d tree); beyond that it is below 4e-6 of its peak.
    w = cloud_km / 6371.0
    m = max(int(4 * 4.0 * np.pi / w**2), 1)
    c = rng.standard_normal((m, 3))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    amp = rng.standard_normal(m)
    field = np.zeros(len(v))
    tree = cKDTree(v)
    for j, hits in enumerate(tree.query_ball_point(c, r=5.0 * w)):
        if hits:
            d2 = np.sum((v[hits] - c[j]) ** 2, axis=1)    # squared chord on the unit sphere
            field[hits] += amp[j] * np.exp(-d2 / (2 * w**2))
    clear = min(1.0, n / len(v))
    keep = field <= np.quantile(field, clear)
    v = v[keep]
    v = v[rng.permutation(len(v))][:n]                    # shuffle; at most n
    return np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2], -1, 1))


def make_obs(n, domain, nranks, seed=0, pattern="uniform", decomposition=None,
             departures="white", c=None, alpha=None, sigma_range=(0.8, 1.2),
             empty_ranks=(), skew=None, n_groups=1, group_layout="random",
             geometry="plane"):
    """Generate n observations scattered at random over nranks initial MPI tasks.

    Parameters
    ----------
    geometry : "plane" (positions in domain) or "sphere" (lon, lat in radians from
        _positions_sphere; domain is ignored; patterns PATTERNS_SPHERE)
    n : number of observations (approximate for pattern="geostationary", and rounded
        down to a multiple of n_groups for group_layout="colocated")
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
    n_groups : number of correlation groups (field "group", 0 .. n_groups-1). Errors
        are correlated only within a group.
    group_layout : how groups are assigned when n_groups > 1
        "random"    -> each observation gets a random group (e.g. different images
                       or time slots at independent positions)
        "colocated" -> n // n_groups positions, each carrying one observation of
                       every group (e.g. the channels of one pixel)

    Returns
    -------
    list of nranks OBS_DTYPE arrays. Global IDs are unique but shuffled and
    non-contiguous (1000 + 7k), so no code can rely on id == index. orig_rank and
    orig_index record each observation's initial position.
    """
    if group_layout not in ("random", "colocated"):
        raise ValueError(f"unknown group_layout {group_layout!r}")
    if n_groups < 1:
        raise ValueError("n_groups must be >= 1")
    if group_layout == "colocated" and n < n_groups:
        raise ValueError("colocated groups need n >= n_groups")
    rng = np.random.default_rng(seed)
    colocated = n_groups > 1 and group_layout == "colocated"
    n_pos = n // n_groups if colocated else n
    sphere = geometry == "sphere"
    grid = geostationary_shape(n_pos, domain) if pattern == "geostationary" and not sphere else None
    if grid is not None:
        n_pos = grid[0] * grid[1]
    disc = sphere and pattern == "geo_disc"
    if disc:                           # the disc decides its own count (at most n_pos)
        x, y = _geo_disc(n_pos, rng)
        n_pos = len(x)
    n = n_pos * n_groups if colocated else n_pos
    obs = empty(n)
    obs["global_id"] = 1000 + 7 * rng.permutation(n).astype(np.int64)
    if not disc:
        if sphere:
            x, y = _positions_sphere(n_pos, pattern, rng)
        else:
            x, y = _positions(n_pos, domain, pattern, rng, decomposition, grid=grid)
    if colocated:
        x, y = np.repeat(x, n_groups), np.repeat(y, n_groups)
        obs["group"] = np.tile(np.arange(n_groups), n_pos)
    elif n_groups > 1:                 # drawn only here, so one group keeps the old stream
        obs["group"] = rng.integers(0, n_groups, n)
    obs["x"], obs["y"] = x, y
    obs["sigma"] = rng.uniform(*sigma_range, n)
    if departures == "white":
        obs["value"] = obs["sigma"] * rng.standard_normal(n)
    elif departures == "correlated":
        R = build_R(obs["x"], obs["y"], obs["sigma"], c, alpha, obs["group"], geometry)
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
