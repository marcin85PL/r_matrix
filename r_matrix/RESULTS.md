# Correlated-R prototype: results

**Status as of 2026-09-30.** These numbers were produced by the code in this directory at that date, using `notebooks/02_experiments.ipynb`. They go stale as soon as the code or the defaults change, so re-run the notebook and update this file together.

How to run the code, and how it works, is in [README.md](README.md). This file covers only what was found.

## 1. Baseline configuration (`Config()`)

| Parameter | Value |
|---|---|
| Domain | 12 × 9, non-periodic, Euclidean distance |
| Decomposition | 4 × 3 = 12 tasks, each subdomain 3 × 3 = 6ℓ × 6ℓ |
| Observations | 3000, uniformly random, about 7 per ℓ²; initial task uniformly random |
| Correlation | Gaspari–Cohn, ℓ = c = 0.5 (support 2c = 1.0), α = 0.8 |
| Departures | d ~ N(0, R), drawn with a global Cholesky, i.e. innovations consistent with R |
| σ | uniform on [0.8, 1.2] |
| Halo | h = 1.0 (2ℓ), "rect" criterion; experiments sweep h/ℓ |

## 2. Results

Unless stated otherwise, the setup is the baseline, and the error is z̃ − z* against the exact global solve. Bytes are those of one application of R⁻¹ as counted by `MockComm`. Wall times come from a single process and are only indicative.

### 2.1 Main result: error against halo width (uniform obs)

| h/ℓ | relative L2 error | max error / max\|z*\| | residual ‖Rz̃−d‖/‖d‖ | peak halo/core | largest local n | halo bytes/apply |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 5.1e-2 | 1.2e-1 | 2.3e-1 | 0.37 | 341 | 5.6 kB |
| 1 | 2.5e-2 | 4.1e-2 | 1.0e-1 | 0.89 | 440 | 12 kB |
| 2 | 6.9e-3 | 1.4e-2 | 2.5e-2 | 1.9 | 696 | 27 kB |
| 3 | 1.7e-3 | 3.3e-3 | 7.1e-3 | 3.2 | 981 | 44 kB |
| **4** | **4.8e-4** | **1.0e-3** | 1.8e-3 | 4.7 | 1267 | 62 kB |
| 5 | 1.2e-4 | 2.1e-4 | 5.3e-4 | 6.4 | 1642 | 83 kB |
| 6 | 2.4e-5 | 6.8e-5 | 9.1e-5 | 8.3 | 2057 | 105 kB |

- **The error decays exponentially, by roughly 3–4× per extra ℓ of halo.** The approximation converges, but a halo equal to the correlation support is nowhere near enough. At h = 2c = 2ℓ, the relative L2 error is still 7e-3.
- The notebook uses placeholder gates: relative L2 ≤ 1e-3, relative max ≤ 1e-2, and peak halo/core ≤ 6. The **smallest passing halo is h ≈ 4ℓ**.
  - There, each task's local problem has about 5× more halo than core observations. The largest local system is about 1270 × 1270, against 250 owned obs.
  - The thresholds are placeholders and must be agreed before a decision (slide 8).
- **Communication is modest and bounded.** Each task talks to at most 8 neighbours, and the halo update is the only part of the recurring traffic that grows with h. Forward and reverse moves are fixed at about 22 kB each per application.
- **The cost is local compute and memory.** Dense Cholesky costs O(n_loc³) at setup and O(n_loc²) per apply, and n_loc grows roughly like (core + 2h)².

### 2.2 Sensitivity

**Correlated fraction α** (relative L2 error; a larger α means a smaller nugget and a worse-conditioned R):

| α | h = 2ℓ | 3ℓ | 4ℓ | 5ℓ | 6ℓ | smallest h/ℓ with error < 1e-3 |
|---|---:|---:|---:|---:|---:|---:|
| 0.5 | 2.0e-3 | 3.5e-4 | 6.2e-5 | 9.8e-6 | 1.2e-6 | ≈ 2.5 |
| 0.8 | 6.9e-3 | 1.7e-3 | 4.8e-4 | 1.2e-4 | 2.4e-5 | ≈ 3.6 |
| 0.95 | 1.7e-2 | 6.0e-3 | 2.6e-3 | 8.2e-4 | 2.6e-4 | ≈ 5 |

**This is the dominant parameter.** How fast R⁻¹ decays is set by the conditioning of R, so the halo needed depends strongly on how much of the error variance is correlated. For real instruments, α should be estimated (e.g. from Desroziers diagnostics) before h is fixed.

**Observation density** (relative L2 error):

| obs per ℓ² | h = 2ℓ | 3ℓ | 4ℓ | 5ℓ | 6ℓ |
|---:|---:|---:|---:|---:|---:|
| 2.3 | 4.4e-3 | 7.5e-4 | 1.2e-4 | 3.0e-5 | 3.1e-6 |
| 6.9 | 6.9e-3 | 1.7e-3 | 4.8e-4 | 1.2e-4 | 2.4e-5 |
| 11.6 | 6.9e-3 | 2.1e-3 | 6.6e-4 | 1.9e-4 | 4.8e-5 |

Denser data decays somewhat more slowly, but the effect is weaker than that of α. The cost, on the other hand, scales directly with density, because n_loc ∝ density × (core + halo) area.

### 2.3 Robustness: decompositions and patterns at h = 3ℓ

| Tasks | pattern | relative L2 error | max error (relative) | owned CV | peak halo/core | largest local n |
|---|---|---:|---:|---:|---:|---:|
| 2×2 | uniform | 1.1e-3 | 2.7e-3 | 0.01 | 0.66 | 1243 |
| 2×2 | clustered | 7.5e-4 | 6.3e-3 | 0.58 | 5.3 | 2132 |
| 4×3 | uniform | 1.7e-3 | 3.3e-3 | 0.05 | 3.2 | 981 |
| 4×3 | clustered | 1.3e-3 | 6.4e-3 | 1.06 | 10.5 | 1717 |
| 4×3 | coincident | 1.6e-3 | 2.8e-3 | 0.07 | 3.6 | 976 |
| 4×3 | boundary | 1.7e-3 | 3.2e-3 | 0.11 | 3.0 | 978 |
| 8×6 | uniform | 2.5e-3 | 3.3e-3 | 0.12 | 10.0 | 551 |
| 8×6 | clustered | 2.2e-3 | 9.2e-3 | 1.97 | 93.5 | 1674 |

- **The accuracy is robust.** Across all patterns and decompositions, the error stays within 0.8–2.7e-3. It grows mildly with the number of tasks, because there are more internal edges.
- **Clustering is a load-balance and cost problem, not an accuracy problem.** On 8×6 one task has a halo 94× its core, and the observation-count CV is about 2. A regular grid decomposition is a poor fit for clustered data. Weighted or recursive-bisection decompositions are the obvious next step.
- Coincident and edge-snapped observations cause no problems. The nugget keeps R positive definite, and the ownership contract is exercised by the tests.

**Halo criterion** (h = 3ℓ):

| Criterion | relative L2 error | total halo |
|---|---:|---:|
| `rect` | 1.7e-3 | 5465 |
| `obs` | 2.1e-3 | 5089 |

`obs` gives a 7 % smaller halo for slightly more error. The brute-force and k-d tree filters give identical ID sets.

### 2.4 Where the error lives: geostationary-like contour maps

With the `geostationary` pattern (3024 pixels, owned CV 0.02), the accuracy matches the uniform case:

| h/ℓ | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| relative L2 error | 2.9e-2 | 9.8e-3 | 2.3e-3 | 6.0e-4 | 1.9e-4 | 4.0e-5 |

Notebook section 6 contours d, z*, z̃ and z̃ − z* over the whole domain.

- **The error is concentrated in bands along the internal subdomain edges.** It is largest where a core observation lies closest to the edge of its own halo, i.e. next to the core boundary. The bands shrink and fade as h grows. In the uniform case at h = 2ℓ (notebook section 5), the median error 3ℓ inside a core is about 45× smaller than next to the edge.
- **The R⁻¹ footprint of a single pixel** shows the mechanism. Near a four-task corner, the exact column of R⁻¹ is a smooth, isotropic, exponentially decaying bump. In the approximation, only the four tasks whose halos contain the pixel ever see it. The approximate column is therefore cut off sharply at the outer edges of those four cores, and it is slightly wrong near the halo edges.
  - A direct consequence: the approximate operator is **not symmetric**, since row and column come from different local problems.
  - This matters if it is used inside a symmetric solver (conjugate gradients) or as a preconditioner.

## 3. Status against the acceptance gates (slide 8)

| Gate | Status |
|---|---|
| REDISTRIBUTION | ✅ Count and ID set preserved, unique owner, boundary and empty-task cases pass. Checked in every run and in the tests. |
| HALO + MAP | ✅ Geometric and k-d tree ID sets identical; handshake count + hash verified; exchanged values equal the owner values. |
| NUMERICAL | ✅ Measured: relative L2, max error and residual against the exact solve (tables above). Pass or fail depends on the thresholds. |
| ROBUSTNESS | ✅ 2×2 / 4×3 / 8×6 × 5 patterns; accuracy stable. ⚠️ Clustered obs break the load balance of a regular decomposition. |
| COST | ✅ Reported: setup and recurring bytes, messages, peers, halo/core ratio, local size and memory. |
| DECISION | ⏳ Thresholds and decision owner still to be agreed. With the placeholders, h ≈ 4ℓ passes at α = 0.8. |

## 4. Limitations and suggested next steps

- **The mock is serial.** Bytes, messages and peers are exact, but times are not parallel timings. The next check is a real MPI (mpi4py) version, which means rewriting the per-rank loops as SPMD code (see the `comm.py` docstring), compared with sparse or neighbourhood exchange (slide 4).
- **Local solves are dense.** At h ≈ 4ℓ, the local R alone takes about 13 MB per task in this small setup (the factor takes as much again), and the cost is O(n_loc³). R_loc is sparse, so a sparse Cholesky (with some fill-in) or a CG solve with a fixed iteration count is the natural next step. That is the local-solver work of slide 8.
- **The operator is non-symmetric.** A symmetrised variant, e.g. ½(Ã + Ãᵀ), which needs one extra halo exchange, or overlapping-Schwarz-style weighting of the halo results, would make the approximate R⁻¹ usable inside CG.
- **Model realism:**
  - add channels, inter-channel correlation and time slots (slide 1 assumptions 1 and 2);
  - add a spherical or periodic domain;
  - add clear-sky gaps (cloud masks) to the geostationary pattern.
- **Decomposition:** weighted or recursive-bisection partitions for clustered observations.
