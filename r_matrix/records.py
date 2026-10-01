"""Observation record layout and input screening (deck slide 2).

Every observation travels between mock MPI tasks as one fixed-size record of the
numpy structured dtype ``OBS_DTYPE``. Because the size is fixed and the fields are
explicitly typed, a record array can be shipped with a single MPI_Alltoallv, which
is how a real implementation would do it (as an MPI derived datatype or raw bytes).

Field meanings
--------------
global_id   int64    Unique, stable identifier of the observation. It is the key used
                     for sorting, deduplication and cross-rank lookups. It is never
                     assumed to be contiguous or equal to an array index.
x, y        float64  Position in the regular 2-D domain.
value       float64  The departure d (the vector that R^-1 is applied to).
sigma       float64  Observation-error standard deviation; R_ii = sigma_i^2.
orig_rank   int32    The rank the observation lived on before redistribution, and
orig_index  int32    its position there. Together they are the return address used
                     when results are sent back to the original layout.
"""
import numpy as np

OBS_DTYPE = np.dtype([
    ("global_id", "i8"),
    ("x", "f8"),
    ("y", "f8"),
    ("value", "f8"),
    ("sigma", "f8"),
    ("orig_rank", "i4"),
    ("orig_index", "i4"),
])


def empty(n=0):
    """Return a zero-initialised record array of length n."""
    return np.zeros(n, dtype=OBS_DTYPE)


def valid_mask(obs, domain):
    """Boolean mask of the records that may enter the pipeline.

    A record is rejected (mask False) if its coordinates are NaN/inf or outside the
    closed domain [xmin, xmax] x [ymin, ymax], if its value is not finite, or if
    sigma <= 0. The screening runs on the initial rank *before* any communication,
    so bad records never travel.
    """
    x, y = obs["x"], obs["y"]
    return (np.isfinite(x) & np.isfinite(y) & np.isfinite(obs["value"]) & (obs["sigma"] > 0)
            & (x >= domain.xmin) & (x <= domain.xmax)
            & (y >= domain.ymin) & (y <= domain.ymax))


def check_unique_ids(obs):
    """Raise ValueError if a record array contains the same global_id twice.

    This is a rank-local check. A duplicate spread over several ranks is caught by
    ``redistribute.check_redistribution``, which plays the role of a global reduction.
    """
    ids = obs["global_id"]
    if np.unique(ids).size != ids.size:
        raise ValueError("duplicate global_id detected")
