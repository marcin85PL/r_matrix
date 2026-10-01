# `r_matrix`: a prototype for applying correlated R⁻¹ on domain-decomposed observations

This is a small Python prototype that tests one idea: in an NWP code where observations are spread over many MPI tasks, can each task apply the inverse of a **correlated** observation-error covariance matrix R using only its own observations plus a **halo** of nearby ones? And how accurate and how expensive is that?

Everything runs in one Python process on **mock MPI tasks**, and every approximate answer is checked against an exact global solve.

| If you want to… | read |
|---|---|
| get it running | [§1 Getting started](#1-getting-started) |
| see the whole pipeline on a toy problem, step by step | `notebooks/01_walkthrough.ipynb` (about 15 min) |
| understand why a halo is needed | [§3 Background](#3-background) |
| know what was found | [RESULTS.md](RESULTS.md) |
| change or extend something | [§7 Extending the code](#7-extending-the-code), [§8 Starter exercises](#8-starter-exercises) |

---

## 1. Getting started

The repository ([github.com/marcin85PL/r_matrix](https://github.com/marcin85PL/r_matrix)) has this layout:

```
r_matrix/                          the repository
  pyproject.toml                   package metadata and dependencies
  r_matrix/                        the package (this README, RESULTS.md)
  tests/test_r_matrix.py           pytest suite
  notebooks/01_walkthrough.ipynb   start here
  notebooks/02_experiments.ipynb   the full experiments behind RESULTS.md
```

```bash
git clone https://github.com/marcin85PL/r_matrix.git && cd r_matrix
python3 -m venv .venv && source .venv/bin/activate      # tested with Python 3.14
pip install -e ".[all]"                                 # editable install, plus test and notebook extras
python -m pytest -q                                     # expect: 26 passed in a few seconds
python -m ipykernel install --user --name r_matrix     # makes the venv available to Jupyter
```

The install is *editable* (`-e`), so `import r_matrix` works from any directory, including `notebooks/`, and your edits to the code take effect without reinstalling. Use `pip install -e .` for the core dependencies only (numpy, scipy), `.[test]` to add pytest, or `.[notebooks]` to add matplotlib, pandas and ipykernel.

Open `notebooks/01_walkthrough.ipynb` and choose the `r_matrix` kernel. The notebooks were saved with a kernel called `env_xarray`, so Jupyter or VS Code will ask you to choose one.

The shortest possible use:

```python
from r_matrix import Config, run
res = run(Config(), h=2.0)          # one full run; any Config field can be overridden as a keyword
print(res.metrics["rel_l2"])        # relative error against the exact global solve
```

`run` takes about 0.5 s with the default 3000 observations. The exact reference solve is dense, so it grows as n³. Keep `n_obs` at or below about 6000, or pass `reference=False`.

## 2. The one convention you must know

**Every per-task quantity is a Python list of length P, and element `r` is what MPI task `r` holds.**

```python
owned[3]          # the observations owned by task 3 (a numpy structured array)
halos[3]          # task 3's halo copies
solvers[3]        # task 3's local solver
```

A *communication* function takes the lists of all tasks at once and returns the lists of all tasks, as if every task had called the MPI routine at the same moment:

```python
recvbufs, recvcounts = comm.alltoallv(sendbufs, sendcounts, "setup:halo:records")
#          ^ list over receivers           ^ list over senders
```

Loops of the form `for r in range(P): ...` are therefore "what every task does locally". In a real MPI code each process would run only its own iteration. `MockComm` (`comm.py`) also counts the bytes and messages that would cross the network, per named *phase*:
- phases whose names start with `setup:` happen once per observation geometry;
- phases whose names start with `recurring:` happen every time R⁻¹ is applied, and are the cost that matters.

## 3. Background

**Why R⁻¹?** In variational data assimilation (3D/4D-Var) the observation term of the cost function is ½ dᵀR⁻¹d. Here d = y − H(x) is the vector of **departures** (observation minus model equivalent), and R is the observation-error covariance. Its gradient needs z = R⁻¹d, at every iteration of the minimisation. With uncorrelated errors, R is diagonal and R⁻¹d is a trivial division by σ². Many modern observations (satellite radiances, high-resolution imagers, Doppler winds) have **spatially correlated** errors, so R has off-diagonal entries.

**The model used here.**

    R_ij = σ_i σ_j [ α ρ(|x_i − x_j|) + (1 − α) δ_ij ]

- ρ is the **Gaspari–Cohn** function. It looks like a Gaussian with length scale ℓ, but it is exactly zero beyond 2ℓ, so R is sparse.
- α is the correlated fraction of the error variance, and 1 − α is an uncorrelated "nugget". The nugget keeps R well-conditioned, even when observations coincide.

**Why the problem is hard.** R is sparse, but **R⁻¹ is dense**. Every observation influences every other one, with a weight that decays roughly exponentially with distance, as the walkthrough's matrix plot shows. A task that owns part of the domain cannot compute its part of R⁻¹d exactly without seeing all observations.

**The approximation tested.** Each task solves R_loc z_loc = d_loc over its own (**core**) observations plus a **halo** of neighbouring observations within distance h. It keeps only the core part of z_loc. Core observations near the edge of their task's region "see" a truncated problem, and that is the error. The error shrinks as h grows, while the cost (halo communication, local matrix size) rises. The prototype measures that trade-off.

**Where the observations start.** In an NWP code observations arrive on MPI tasks in essentially random order with respect to location, for example in file order. So they must first be **redistributed** to the task that owns their location, and at the end the results must be sent **back**.

**Further reading:**
- Gaspari & Cohn (1999), QJRMS 125, 723–757: the compactly supported correlation functions.
- Weston, Bell & Eyre (2014), QJRMS 140, 2420–2429: an operational example of correlated errors (IASI inter-channel).
- Janjić et al. (2018), QJRMS 144, 1257–1278: sources and representation of observation errors.
- Desroziers et al. (2005), QJRMS 131, 3385–3396: diagnosing R from departures.
- MPI collectives: the MPI standard (mpi-forum.org), chapter "Collective Communication", for `MPI_Alltoall` and `MPI_Alltoallv`.

## 4. Notation

| Symbol | Code | Unit | Default | Meaning |
|---|---|---|---|---|
| P = Pₓ × Pᵧ | `px`, `py`, `decomp.size` | – | 4 × 3 = 12 | number of mock MPI tasks, one subdomain (cell) each |
| ℓ | `c` | domain units | 0.5 | Gaspari–Cohn length scale. The code calls it `c`, after the paper; the text and plots call it ℓ. **They are the same.** |
| 2ℓ | `2 * c` | domain units | 1.0 | correlation support: ρ = 0 beyond it |
| α | `alpha` | – | 0.8 | correlated fraction of the error variance |
| h | `h` | **domain units** | 1.0 | halo width. Results are quoted as **h/ℓ**, so `run(h=2.0)` means h = 4ℓ. |
| d | field `value` | – | – | departures, the vector R⁻¹ is applied to |
| z*, z̃ | `z_ref_orig`, `z_orig` | – | – | exact and approximate R⁻¹d |
| core / owned | `owned[r]` | – | – | observations inside task r's cell |
| halo | `halos[r]` | – | – | copies of other tasks' observations within h of cell r |

## 5. How the code is organised

### Pipeline

`pipeline.run` executes these steps in order. The walkthrough notebook does the same by hand.

| Step | Deck slide | Module | What happens |
|---|---|---|---|
| 0 Generate | 2 | `generate.py` | seeded observations on **random** initial tasks |
| 1 Ownership | 3 | `decomposition.py` | `owner = i + Pₓ·j` for a regular Pₓ × Pᵧ grid of cells |
| 2 Redistribute | 4 | `redistribute.py` | records move to their owners (Alltoall counts + Alltoallv records) |
| 3 Halo discovery | 5 | `halo.py` | owners push nearby records to neighbours, and receivers filter exactly |
| 4 Freeze the map | 7 | `commmap.py` | request IDs → owner translates → confirm count + hash |
| 5 Factorise | 1 | `covariance.py` | `LocalSolver`: dense Cholesky of R over core + halo |
| Recurring apply | 1, 7 | `pipeline.py` | `RunResult.apply_Rinv`: forward d → halo update → local solve → reverse z |
| Validate | 8 | `reference.py`, `diagnostics.py` | exact global solve, error norms, residuals, costs |

### Data structures (glossary)

| Name | Where | What it is |
|---|---|---|
| `OBS_DTYPE` | `records.py` | fixed-size record: `global_id`, `x`, `y`, `value`, `sigma`, `orig_rank`, `orig_index` |
| `global_id` | – | unique, **non-contiguous** observation ID; the key for every lookup |
| `owned[r]` | `redistribute` | task r's observations, **sorted by global_id**; the position is the `owned_local_index` |
| `RedistributionPlan` | `redistribute.py` | frozen routing original ↔ spatial, for moving bare values later |
| `halos[r]`, `halo_owner[r]` | `halo.py` | halo copies sorted by (owner, global_id), and the owner of each |
| `RankMap` | `commmap.py` | `recv[owner]` = slice of `halos[r]`; `send[dest]` = indices into `owned[r]` |
| `LocalSolver` | `covariance.py` | R_loc over [core; halo], factorised once; `apply`, `matvec_core` |
| `RunResult` | `pipeline.py` | everything a run produced, plus `metrics` and `apply_Rinv` |
| phase names | `comm.py` | `setup:*` (once), `recurring:*` (every apply), `diag:*` (validation only) |

### Key design decisions

- **Screen before communicating.** NaN, out-of-domain, non-finite and σ ≤ 0 records are rejected on their initial task and come back as NaN. Empty tasks are valid.
- **Ownership contract.** Cells are `[low, high)`, and only x = xmax / y = ymax are clamped into the last cell, so every point has exactly one owner.
- **Deterministic order everywhere.** Arrays are sorted by global_id (owned) or by (owner, global_id) (halo). Results never depend on message arrival order.
- **Setup versus recurring.** Setup moves full 48-byte records once and freezes the plan and the maps. Each later apply moves only float64 values.
- **Halo width is its own parameter**, separate from the correlation support.
  - The default criterion `rect` takes points within h of the task's cell rectangle.
  - The alternative `obs` takes points within h of any owned observation.
  - Both use the same geometric candidate push. For `obs`, a brute-force and a k-d tree filter are both implemented and must agree.
- **Fail loudly at setup.** The map handshake checks counts and ID hashes, and `check_redistribution` checks counts, the ID set, uniqueness, and that every point is inside its owner's cell.
- **Two residual checks.** ‖Rz̃ − d‖/‖d‖ is computed with the global R, and also distributed via one extra halo exchange (exact when h ≥ 2ℓ).

### Observation patterns (`Config.pattern`)

| Pattern | Layout | Stresses |
|---|---|---|
| `uniform` | i.i.d. uniform | baseline |
| `clustered` | 80 % in 6 tight blobs | load imbalance, very large halos |
| `coincident` | 20 % share another's location | conditioning |
| `boundary` | 20 % exactly on cell edges and xmax/ymax | the ownership rule |
| `geostationary` | jittered regular pixel grid | near-uniform imager coverage; allows contour maps |

### Tests

`tests/test_r_matrix.py` has 26 short tests, one or two per stage plus end-to-end numerical checks. They are also the best usage examples. The numerical checks that matter most:
- with one task (P = 1) the result equals the global solve;
- with a halo covering the whole domain, the result equals the global solve exactly;
- the error decreases as h grows;
- the distributed matvec equals the global one when h ≥ 2ℓ.

## 6. Where to look first

1. `notebooks/01_walkthrough.ipynb`: all stages on 50 observations, printing every data structure.
2. `pipeline.run`: the whole algorithm on one screen. Read its step comments.
3. Whichever module you need to change. Each module's docstring explains its step before the code starts.

## 7. Extending the code

Add a test for anything new in `tests/test_r_matrix.py`, and run `python -m pytest -q` before and after each change.

**A new observation pattern.**
1. Add its name to `PATTERNS` and a branch to `_positions` in `generate.py`. Document it in the module docstring.
2. Add it to the `pattern` parametrize lists in the tests. The redistribution invariants and the robustness test then run on it automatically.

**A new correlation function** (e.g. SOAR, or a Schur product with a taper).
1. `covariance.build_R` is the single place where R is built. It is used by `LocalSolver`, `reference.global_solve` and `generate.make_obs` (correlated departures).
2. Add a `Config` field for the model and pass it through `run` to those three call sites.
3. Keep the support finite. The halo criteria and the "distributed matvec is exact when h ≥ support" test assume ρ = 0 beyond a known distance.

**A different local solver** (sparse Cholesky, CG, …).
1. Write a class with the same interface as `LocalSolver`: the constructor `(core, halo, c, alpha)`, `apply(d_core, d_halo)`, `matvec_core(z_core, z_halo)`, and the attributes `ncore`, `nhalo`, `n`, `nbytes`.
2. Swap it in `pipeline.run`, which is the one place solvers are created. Nothing else needs to change.

**A new metric.** Compute it in `diagnostics.py`, then add it to the `m` dict in `pipeline.run`. `sweep` automatically includes every scalar metric.

**A different decomposition** (weighted, recursive bisection). The pipeline uses `Decomposition` only through `size`, `domain`, `owner`, `cell_rect`, `ranks_intersecting` and `rect_distance`. The `boundary` pattern in `generate.py` also reads `px`, `py`, `dx` and `dy`. A new class providing these works with the rest of the pipeline, but it must still give each task a rectangular cell, because the halo code relies on that.

**Real MPI.** The data layouts and communication calls map directly onto `MPI_Alltoall(v)`. The per-task `for r in range(P)` loops must become SPMD code, where each process runs only its own `r`. See the docstring of `comm.py`.

## 8. Starter exercises

They are roughly in order of difficulty. Each says where to work and how to check the result.

1. **Warm-up: halo width.** In the walkthrough, change `H` to 0.5, 1.0 and 3.0. Tabulate the halo sizes and the relative error. Explain why `H = 3.0` gives about 1e-15. *Check:* the error decreases monotonically.
2. **Conditioning.** For α ∈ {0.3, 0.5, 0.8, 0.95, 0.99}, compute `np.linalg.cond` of the global R (use 500 obs) and the h/ℓ needed for a relative error below 1e-3. Plot one against the other. *Why it matters:* this is what sets the halo width for a real instrument. See section 2.2 of RESULTS.md.
3. **A new pattern: clouds.** Add a `geostationary_cloudy` pattern that removes pixels under a smooth random "cloud" field (e.g. a thresholded, smoothed noise field). Does the error change, or only the cost? *Where:* `generate.py`, plus the tests.
4. **A new correlation function.** Implement a SOAR correlation multiplied by a Gaspari–Cohn taper (a Schur product), as a `Config` option. Test that R stays positive definite and that its support is finite. Compare the error curves with plain GC at the same ℓ. *Where:* `covariance.py`, `pipeline.py`, tests.
5. **Is the approximate operator symmetric?** On a 300-observation problem, build the full approximate matrix Ã column by column with `res.apply_Rinv` on unit vectors. Measure ‖Ã − Ãᵀ‖/‖Ã‖ as a function of h. Then propose and implement a symmetric variant. *Why it matters:* conjugate-gradient minimisers need a symmetric operator.
6. **A sparse local solver.** Replace the dense Cholesky in `LocalSolver` with a sparse one, or with CG using a fixed number of iterations. Compare memory, setup time and apply time with the dense version as `n_loc` grows (h/ℓ from 2 to 6). *Where:* a new class with the `LocalSolver` interface; see §7.
7. **Advanced: a better decomposition for clustered data.** Implement recursive coordinate bisection that balances observation counts, with rectangular cells of unequal size, behind the `Decomposition` interface. Compare the load balance (`metrics["owned"]["cv"]`) and the peak halo/core ratio with the regular grid for `pattern="clustered"`.
8. **Advanced: real MPI.** Port the redistribution step (`redistribute.redistribute` and `RedistributionPlan`) to mpi4py. Run it with `mpirun -n 4`, and check that the per-phase byte counts equal `MockComm`'s.

## 9. File map

```
r_matrix/              the repository
  README.md            repository overview
  LICENSE              Apache License 2.0
  pyproject.toml       package metadata, dependencies, pytest settings
  r_matrix/            the package
    README.md          this guide
    RESULTS.md         findings (dated; regenerate with notebooks/02_experiments.ipynb)
    __init__.py        public API: Config, run, sweep, RunResult, MockComm, Domain, Decomposition
    records.py         OBS_DTYPE, input screening, ID uniqueness
    generate.py        seeded mock observations (5 patterns), random initial tasks
    decomposition.py   Domain, Px × Py Decomposition, owner(), cell geometry
    comm.py            MockComm (alltoall / alltoallv / neighbour exchange) + CommStats
    redistribute.py    setup redistribution (steps A–E), RedistributionPlan (forward/reverse)
    halo.py            geometric candidate push + exact filter ("rect" / "obs", brute / k-d tree)
    commmap.py         RankMap, setup handshake, recurring halo_exchange
    covariance.py      Gaspari–Cohn, build_R, LocalSolver (Cholesky on core + halo)
    reference.py       global exact solve
    diagnostics.py     error metrics, load balance, halo summary
    pipeline.py        Config, run(), RunResult (apply_Rinv, apply_Rinv_exact), sweep()
  tests/test_r_matrix.py           26 pytest tests
  notebooks/01_walkthrough.ipynb   toy problem, step by step
  notebooks/02_experiments.ipynb   full experiments (sections 1–6)
```
