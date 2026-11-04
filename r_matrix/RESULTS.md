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

### 2.5 The symmetrised operator S = ½(Ã + Ãᵀ) (added 2026-11-04)

`Config(symmetric=True)`. Each local solve takes a second right-hand side, [d_core; 0], on the same factor, and the halo part of that solution is summed back into its owners by an adjoint halo exchange (`commmap.halo_exchange_adjoint`). Script: `experiments/symmetric_operator.py`; raw output: `experiments/symmetric_operator.out`. Baseline setup otherwise (3000 obs, 4 × 3 tasks).

- *Generalised eigenvalues* are those of S x = λ R⁻¹ x. If all are close to 1, S is close to R⁻¹ in the energy norm, so the Jo Hessian, and with it the minimisation, is barely changed.
- *Errors* are the relative L2 error of Ã d and S d against R⁻¹ d.
- *Bytes* are all recurring traffic per application.

| α | h/ℓ | ‖Ã−Ãᵀ‖/‖Ã‖ | λ_min(S) | λ_min(R⁻¹) | gen. eig. min | gen. eig. max | error Ã | error S | bytes Ã | bytes S |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 2 | 2.4e-3 | 0.108 | 0.108 | 0.9837 | 1.0163 | 2.0e-3 | 1.7e-3 | 71 kB | 98 kB |
| 0.5 | 4 | 6.3e-5 | 0.108 | 0.108 | 0.9995 | 1.0005 | 6.2e-5 | 4.6e-5 | 106 kB | 168 kB |
| 0.8 | 1 | 2.5e-2 | 0.0703 | 0.0703 | 0.7842 | 1.2397 | 2.5e-2 | 2.4e-2 | 56 kB | 68 kB |
| 0.8 | 2 | 7.5e-3 | 0.0703 | 0.0703 | 0.9368 | 1.0642 | 6.9e-3 | 6.7e-3 | 71 kB | 98 kB |
| 0.8 | 3 | 1.9e-3 | 0.0703 | 0.0703 | 0.9814 | 1.0189 | 1.7e-3 | 1.8e-3 | 88 kB | 131 kB |
| **0.8** | **4** | 4.9e-4 | 0.0703 | 0.0703 | **0.9954** | **1.0046** | 4.8e-4 | **4.1e-4** | 106 kB | 168 kB |
| 0.8 | 6 | 1.8e-5 | 0.0703 | 0.0703 | 0.9996 | 1.0004 | 2.4e-5 | 2.5e-5 | 150 kB | 255 kB |
| 0.95 | 1 | 3.7e-2 | 0.0566 | 0.0599 | 0.4606 | 1.5969 | 4.4e-2 | 5.6e-2 | 56 kB | 68 kB |
| 0.95 | 2 | 1.7e-2 | 0.0599 | 0.0599 | 0.7725 | 1.2404 | 1.7e-2 | 2.2e-2 | 71 kB | 98 kB |
| 0.95 | 4 | 2.4e-3 | 0.0599 | 0.0599 | 0.9681 | 1.0324 | 2.6e-3 | 3.2e-3 | 106 kB | 168 kB |
| 0.95 | 6 | 1.9e-4 | 0.0599 | 0.0599 | 0.9952 | 1.0048 | 2.6e-4 | 3.6e-4 | 150 kB | 255 kB |

Robustness at α = 0.95, the worst-conditioned case:

| tasks | pattern | h/ℓ | λ_min(S) | gen. eig. min | gen. eig. max | error Ã | error S |
|---|---|---:|---:|---:|---:|---:|---:|
| 4×3 | clustered | 2 | 0.0053 | 0.5032 | 1.5208 | 6.7e-3 | 1.9e-2 |
| 4×3 | clustered | 4 | 0.0054 | 0.9492 | 1.0512 | 6.7e-4 | 1.6e-3 |
| 4×3 | coincident | 4 | 0.0576 | 0.9643 | 1.0363 | 1.9e-3 | 2.7e-3 |
| 4×3 | boundary | 4 | 0.0447 | 0.9549 | 1.0453 | 2.0e-3 | 3.2e-3 |
| 4×3 | geostationary | 4 | 0.0747 | 0.9545 | 1.0463 | 3.7e-3 | 5.0e-3 |
| 8×6 | uniform | 4 | 0.0599 | 0.9624 | 1.0382 | 3.7e-3 | 4.8e-3 |
| 8×6 | clustered | 4 | 0.0054 | 0.9338 | 1.0663 | 1.0e-3 | 1.9e-3 |

The full sweep (h/ℓ = 1–6, and h/ℓ = 2 for every robustness case) is in the `.out` file.

- **S is positive definite in all 30 cases.** This includes h = 1ℓ at α = 0.95, where λ_min(S) = 0.057 against 0.060 for R⁻¹.
  - The script computes these eigenvalues on ½(Ã + Ãᵀ) assembled from the local factors, so it is symmetric by construction.
  - The tests check that the operator applied by `apply_Rinv(symmetric=True)` equals this matrix to 1e-12 and is symmetric to 1e-13. They do this on small problems, including clustered data with ranks that have no core and a zero halo.
- **S is close to R⁻¹ in the energy norm.** The spread of the generalised eigenvalues around 1 scales with the asymmetry, at about 10× its value: ±0.5 % at α = 0.8 with h = 4ℓ, ±3 % at α = 0.95, and ±5–7 % for clustered data. It shrinks to 0 as h grows.
- **The accuracy is unchanged or slightly better for α ≤ 0.8. It is 1.2–3× worse at α = 0.95 and for clustered data.** The transpose brings in columns from halo-truncated problems. At α = 0.95 an error below 1e-3 needs h ≈ 5.5ℓ instead of 5ℓ.
- **Cost:** +20–70 % recurring traffic, growing with h/ℓ (+21 % at 1ℓ, +38 % at 2ℓ, +58 % at 4ℓ, +70 % at 6ℓ), because the halo is exchanged twice; the forward and reverse moves do not change. The second right-hand side reuses the factor, so the setup cost does not change.
- **Verdict for the IFS design (plan decision D3):** the symmetrised core + halo operator is usable inside CG, with Jo and its gradient consistent. Choose the halo width from the error gate, which is the slightly stricter requirement for S.

### 2.6 Correlation groups (added 2026-11-04)

`Config(n_groups=G, group_layout="colocated" | "random")`. Each observation has a `group` (in the IFS: channel × time slot). R_ij = 0 between groups, so R is block diagonal. On each rank, every group that has core observations there gets its own local R block and its own factor. Halo observations of other groups are skipped. No rank stores an n × n matrix for all groups together. Script: `experiments/groups.py`; raw output: `experiments/groups.out`.

- **Correctness (tests):**
  - Two groups give the same result as two separate one-group runs, to 1e-13, for both layouts, plain and symmetrised. This exact equality holds for the geometric `rect` halo criterion. With `obs`, the combined halo is a superset, so results differ at the level of the halo truncation error.
  - The symmetrised operator with groups equals ½(Ã + Ãᵀ), including ranks with no core observations.
  - The block-wise matvec is exact, so the distributed and global residuals agree.
  - The operator is exact when the halo covers the domain.
  - Skipped halo groups give exact zeros.
- **One group is bit-identical to the code before groups were added.** Checked by hand, not in the test suite, on 15 034 values over 5 configurations, including the residual and memory metrics. Only `bytes_setup` changes: about +7 %, because the record grew from 48 to 52 bytes to carry `group`.
- **Cost at a fixed total of 3000 observations** (symmetrised, α = 0.8, h = 4ℓ; setup time from a single process, indicative):

| groups | layout | rel. L2 error | largest local n (all groups) | largest local R memory | bytes / apply | setup time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | – | 4.1e-4 | 1267 | 12.8 MB | 168 kB | 0.44 s |
| 2 | colocated | 2.0e-4 | 1286 | 6.6 MB | 169 kB | 0.19 s |
| 4 | colocated | 6.4e-5 | 1372 | 3.8 MB | 173 kB | 0.10 s |
| 8 | colocated | 4.7e-5 | 1336 | 1.8 MB | 172 kB | 0.06 s |
| 2 | random | 2.0e-4 | 1267 | 6.4 MB | 168 kB | 0.18 s |
| 4 | random | 7.5e-5 | 1267 | 3.2 MB | 168 kB | 0.09 s |
| 8 | random | 1.9e-5 | 1267 | 1.6 MB | 168 kB | 0.06 s |

- **Memory** (local R blocks; the factors take as much again) is Σ n_g². It falls exactly as 1/G when groups are equal in size (random: 6.42, 3.21, 1.61 MB), and roughly so otherwise (colocated: 7.2× at G = 8). Setup time falls about as 1/G as well. Traffic does not change, because the same values move.
- **Error.** The error falls as G grows. A likely cause is that each group is sparser (cf. §2.2), but the experiment does not isolate it: at G = 8, colocated and random give different errors (4.7e-5 and 1.9e-5).
- **The implication for the IFS** is the reverse reading: adding channels or time slots as groups adds independent blocks. Memory therefore grows **linearly** with the number of groups, and so does factorisation cost, where one joint matrix would grow quadratically and cubically. The halo width is set per group by its own α and density.
- **The halo search is still group-blind.** The halo of a rank contains nearby observations of all groups, and those of groups without core observations are skipped in the solve. A group-aware halo would save traffic only when the groups' positions differ a lot.

### 2.7 The sphere (added 2026-11-04)

`Config(geometry="sphere")`. Positions are (lon, lat) in radians, as in the IFS ODB. Longitudes outside [−π, π) are wrapped and remain valid. Distances are chords on the Earth sphere, in km, and GC of the chordal distance stays positive definite (`geometry.py`).

**Decomposition.** `SphereDecomposition(px, py)`: py equal-area latitude bands (uniform in sin φ) × px longitude sectors, numbered as on the plane. It gives the **exact** chordal distance from a point to a cell, in closed form over the cell's two parallels and two meridian arcs. The geometric `rect` halo is therefore exact everywhere, including across 180° and around the poles. The `obs` halo uses a 3-D k-d tree.

**Tests** (26 new):
- ownership at 180°E/W, at 2π and at the poles;
- equal-area cells;
- the exact cell distance against dense sampling, never larger than the distance to any cell point. This covers polar wedges, a middle cell, cells π and 2π wide (px = 2, 1), a single band (py = 1), and points exactly on sector edges or at the poles;
- redistribution with no rejections, for unwrapped longitudes too;
- both halo criteria against brute force, and the k-d tree against brute force;
- exactness with one task or with a halo larger than the sphere;
- a matvec that is exact at h = 2ℓ across poles and dateline;
- the error falling with h;
- the exact solution unchanged when longitudes are shifted (so unwrapped longitudes are accepted);
- the `rect` halo contains the `obs` halo for the same h;
- groups on the sphere.

**The plane is unchanged:** bit-identical to the code before the sphere on 21 059 values over 7 configurations (both halo criteria, k-d tree, groups, setup and halo byte counts). Checked by hand.

**Accuracy and cost** (`experiments/sphere.py`, raw output `experiments/sphere.out`). 3000 obs on the whole sphere, ℓ = c = 1000 km (5.9 obs per ℓ², against 6.9 on the plane), symmetrised, α = 0.8:

| geometry | tasks | pattern | h/ℓ = 2 | 3 | 4 | 5 | 6 | peak halo/core at 4ℓ | bytes/apply at 4ℓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| plane | 4×3 | uniform | 6.7e-3 | 1.8e-3 | 4.1e-4 | 1.1e-4 | 2.5e-5 | 4.7 | 168 kB |
| sphere | 4×3 | uniform | 7.0e-3 | 1.7e-3 | 3.7e-4 | 7.9e-5 | 1.6e-5 | 4.0 | 202 kB |
| sphere | 4×3 | poles_dateline | 8.5e-3 | 2.0e-3 | 4.6e-4 | 1.1e-4 | 2.7e-5 | 4.8 | 203 kB |
| sphere | 8×6 | uniform | 1.1e-2 | 2.6e-3 | 5.9e-4 | 1.3e-4 | 2.7e-5 | 13.8 | 506 kB |

- **The approximation behaves on the sphere as on the plane.** The error falls by about 4× per ℓ of halo (4.1–4.9× on the sphere, 3.7–4.4× on the plane in this run), and h ≈ 4ℓ gives about 4e-4 at α = 0.8. The distributed matvec equals the global one for every h ≥ 2ℓ, in every case.
- **The poles and the dateline cost a little accuracy.** The special pattern packs extra points near the poles, which raises the local density and the halo/core ratio.
- **The polar wedges are the weak point of the band × sector partition.** With 8×6 tasks, the 16 polar-band cells (8 per pole) are thin wedges meeting at the poles. The peak halo/core ratio reaches 13.8 at 4ℓ and 24 at 6ℓ, and traffic is 2.5× that of 4×3. Equal-regions partitions with polar caps, or density-balanced patches, should do better; that is the comparison of Phase A step 5b.

### 2.8 R-patches separate from MPI tasks: the reproducible mode (added 2026-11-04)

`Config(px, py, n_tasks=T, patch_map="block" | "round_robin")`. The px × py decomposition defines **R-patches**, and every stage runs per patch. Patches are mapped to T tasks deterministically, several per task. The observations of a task in the original layout sit at its home patch (its lowest-numbered one). `MockComm` counts traffic between tasks only, since traffic between patches of one task is free. `n_tasks=None` keeps one patch per task.

- **The answer depends only on the patch layout.** For a fixed layout, z is **bit-identical** for T = 1, 4, 5 (round-robin), 12, 24 and one patch per task. The error metrics are too, because they are reduced in global-ID order. This was checked on the plane, with 3 colocated groups on clustered data, and on the sphere with poles and dateline (tests). The initial placement of observations differs with T in every case.
  - Nothing had to be made deterministic for this. All orderings were already keyed on (patch, global ID): owned data sorted by ID, halos by (owner patch, ID), adjoint sums in source-patch order, one factor per (patch, group).
  - It also relies on the dense BLAS/LAPACK kernels giving identical results for identical matrices, which holds in this single-process mock.
  - This is the prototype of plan decision D2's flag: **fix the patch layout, and the answer no longer depends on NPROC**.
- `n_tasks=None` is bit-identical to the code before this change, on 21 059 values over 7 configurations. Checked by hand.

**Cost of the reproducible mode** (`experiments/patches.py`, raw output `experiments/patches.out`; 3000 obs on the plane, symmetrised, α = 0.8, h = 4ℓ). It compares a fixed layout of 8 × 6 = 48 patches with the relaxed mode, one patch per task:

| tasks | mode | patches | rel. L2 error | largest patch n | largest task memory | bytes / apply | messages / apply | dedup saving: halo update | dedup saving: whole apply |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | relaxed | 4 | 2.8e-4 | 1436 | 16.5 MB | 79 kB | 48 | – | – |
| 4 | reproducible | 48 | 6.8e-4 | 759 | 41.6 MB | 233 kB | 44 | 67 % | 28 % |
| 12 | relaxed | 12 | 4.1e-4 | 1267 | 12.8 MB | 168 kB | 380 | – | – |
| 12 | reproducible | 48 | 6.8e-4 | 759 | 14.9 MB | 350 kB | 432 | 53 % | 23 % |
| 48 | either | 48 | 6.8e-4 | 759 | 4.6 MB | 440 kB | 4586 | 0 % | 0 % |

Bytes count every patch-to-patch transfer between different tasks. Messages are one per non-empty task pair, i.e. all data between two tasks packed into one message.

- **Accuracy follows the patch layout, not the task count.** More patches mean more internal edges. The halo must be sized for the fixed layout.
- **Overhead when few tasks hold many patches.** A task stores the halos of all its patches, and these overlap: 2.5× the memory of relaxed mode at 4 tasks. Each halo value is also sent once per patch that needs it.
  - Sending each value of the forward halo update **once per task** would save the bytes in the last two columns: 67 % of that update and 28 % of a whole application at 4 tasks, 53 % and 23 % at 12. The script computes these and checks them against the recorded halo bytes.
  - The forward and reverse moves cannot be reduced this way.
  - **The adjoint halo exchange of the symmetric operator is left per patch.** Pre-summing it per task would change the floating-point summation order and break the bit-identity. Deduplicating it too would need an order-independent (reproducible) summation, such as the IFS's `ORDER_INDEP_GLOBAL_SUM`.
  - The mock does not deduplicate. Its halo bytes are therefore upper bounds for a task-level implementation; its forward and reverse bytes are exact.
- **Implication for the library:**
  - build the forward halo map at task level (the union of the task's patch halos), and scatter locally to the patches. Keep the adjoint per patch, or sum it in an order-independent way;
  - choose the fixed patch count for the largest task count that will be used, accepting the overhead at smaller counts;
  - or offer the relaxed mode when reproducibility is not required.

### 2.9 Realistic SEVIRI images (added 2026-11-04)

Sphere pattern `geo_disc`. It is a jittered near-uniform lattice within 66° of the sub-satellite point, with cloud gaps from a smooth random field (systems of about 500 km). It is calibrated on SEVIRI clear-sky radiances in a real IFS CCMA (experiment jdfp, 2022060100; plan repository, `plan/ifs_deep_dive/05_seviri_ccma_quicklook.md`):
- **Every channel has the same pixel spacing** in the CCMA (median nearest-neighbour distance 54–55 km). Channels with fewer active pixels have more of the disc masked, not wider spacing. So the lattice is fixed, and the number of pixels sets the clear fraction. The generator gives 43 / 56 / 67 km (10th / 50th / 90th percentile) for any image size; the CCMA has 49 / 55 / 79 km.
- **Active pixels per hourly image:** Meteosat-11 ch 5 / 6 / 9 = 10 680 / 5 297 / 991; Meteosat-8 ch 5 / 6 = 8 904 / 4 939. With 12 images, that is 60 groups and about 370 000 data per 12-hour window.

**The error parameters are provisional:** ℓ = c = 60 km and α ∈ {0.5, 0.8}, from a rough look at first-guess departure correlations (0.56 below 50 km, 0.3 at 50–100 km, about 0 beyond 150 km, water-vapour channels). Part C brackets ℓ up to 150 km.

Script: `experiments/seviri.py`; raw output: `experiments/seviri.out`. Symmetrised operator, global equal-area patches. Errors are the median (and range) over three seeds, which vary the cloud pattern, the jitter and the departures.

**A. Accuracy, channel-6-sized image** (5 300 pixels, 192 patches):

| α | h/ℓ = 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 1.8e-4 | 1.4e-5 | 7.2e-7 | 4.8e-8 | 3.8e-9 |
| 0.8 | 6.7e-4 (6.4–8.2e-4) | 7.6e-5 | 6.8e-6 | 8.0e-7 | 9.1e-8 |

**C. Sensitivity to ℓ and image size** (α = 0.8, 192 patches; median, range over three seeds):

| pixels | ℓ (km) | h/ℓ = 2 | 3 | 4 | largest local n at 4ℓ |
|---:|---:|---:|---:|---:|---:|
| 11 000 (ch 5) | 60 | 7.8e-4 (6.6–8.0e-4) | 8.1e-5 | 9.1e-6 | 864 |
| 5 300 (ch 6) | 100 | 4.7e-3 | 8.7e-4 (8.5–9.2e-4) | 1.6e-4 | 781 |
| 11 000 (ch 5) | 100 | 4.8e-3 | 9.0e-4 (8.9–9.5e-4) | 1.7e-4 | 1 107 |
| 5 300 (ch 6) | 150 | 6.0e-3 | 1.2e-3 (1.1–1.2e-3) | 2.6e-4 | 965 |
| 11 000 (ch 5) | 150 | 6.1e-3 | 1.3e-3 (1.3–1.3e-3) | 2.9e-4 (2.9–2.9e-4) | 1 410 |

**B. Cost of one channel-5-sized image** (11 000 pixels, α = 0.8, h = 4ℓ = 240 km):

| patches | with data | CV of core sizes (patches with data) | largest local n | largest local R | sum of local R | Σ n³/3 | bytes / apply |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | 19 | 0.63 | 1 717 | 24 MB | 123 MB | 6.4 Gflop | 230 kB |
| 192 | 55 | 0.72 | 825 | 5.4 MB | 68 MB | 1.6 Gflop | 293 kB |
| 768 | 165 | 0.70 | 352 | 1.0 MB | 48 MB | 0.5 Gflop | 441 kB |

**D. One 12-hour window** (60 groups, 192 patches, ℓ = 60 km, h = 4ℓ). Sum over all patches; the factors take as much memory again:

| group (× 12 images) | pixels per image | local R per window | factorisation per window |
|---|---:|---:|---:|
| Met-11 ch 5 | 10 680 | 0.78 GB | 17.9 Gflop |
| Met-11 ch 6 | 5 297 | 0.27 GB | 4.9 Gflop |
| Met-11 ch 9 | 991 | 0.03 GB | 0.3 Gflop |
| Met-8 ch 5 | 8 904 | 0.59 GB | 12.5 Gflop |
| Met-8 ch 6 | 4 939 | 0.25 GB | 4.4 Gflop |
| **total** | 369 732 data | **1.92 GB** (3.8 GB with factors) | **40 Gflop** |

The Met-8 groups are modelled on the same disc as Met-11 (sub-satellite longitude 0°); only their size matters here.

- **Accuracy is set by the local density, which is the same for every channel** (pixels about 55 km apart). Channel-5 and channel-6 images give the same errors at the same h/ℓ.
  - With ℓ ≈ the pixel spacing, there are 1–2 pixels per ℓ². R⁻¹ decays faster than in the toy baseline (7 per ℓ²).
- **Halo needed for a 1e-3 error gate** (α = 0.8):
  - **2ℓ (120 km) at ℓ = 60 km**: 6.4–8.2e-4 over the seeds and both image sizes, a modest margin;
  - **4ℓ (400 km) at ℓ = 100 km**: 3ℓ is marginal at 8.5–9.5e-4;
  - **4ℓ (600 km) at ℓ = 150 km**: 3ℓ fails at 1.1–1.3e-3.
  - So the halo in km must follow the estimated ℓ (plan Phase B step 6). Since α = 0.5 is easier than 0.8 (A), these are on the safe side if α ≤ 0.8.
- **The cost is small.**
  - A whole 12-hour window needs about 1.9 GB of local R (3.8 GB with factors) and 40 Gflop of factorisation, **summed over all tasks**, at ℓ = 60 km and h = 4ℓ.
  - The largest local system is about 830–860 data with 192 patches.
  - Each application then costs about two triangular solves per block: of the order of 0.5 Gflop in total per application of the symmetrised operator.
  - Larger ℓ raises the local n moderately (1 410 at ℓ = 150 km, h = 4ℓ, 192 patches).
- **A global equal-area partition fits SEVIRI poorly.** Only 19 of 48, 55 of 192 and 165 of 768 patches hold data. Among those, the core sizes vary with a CV of 0.6–0.7, and edge and cloud-gap patches have tiny cores with large halo/core ratios. Density-balanced patches are the subject of Phase A step 5b.
- The setup time at 768 patches (93 s) is an artefact of the toy's sphere push, which tests every record against every patch (see `SphereDecomposition.candidate_ranks`).

### 2.10 Equal-area bands against density-balanced patches (added 2026-11-04)

`Config(partition="bisection", criterion="obs")`: `BisectionDecomposition` splits the observations by recursive coordinate bisection in 3-D, along the axis of largest extent, near the count quantile. Any number of patches is possible.

**Ties.** Ownership is a function of position only, so a cut cannot separate points with equal coordinates. Two rules handle this:
- the coordinates are first given a fixed, generic rotation, so that regular grids do not tie on the cut axes;
- each cut is moved to the nearest boundary between distinct values.

Without co-located records every patch then gets the same number of observations (±1; tested on a 40 × 40 latitude–longitude grid). Co-located records (e.g. the channels of one pixel) always share a patch; the balance is then limited by the number of distinct positions (tested).

**Other properties.**
- The partition depends only on the data, not on the number of tasks, so it is compatible with the reproducible mode (tested bit-identical for 1, 3, 5 and 12 tasks).
- Patches are irregular, so the halo is "within h of a core observation" (`obs`). The push uses a bounding cap per patch, a superset (tested directly and against brute force).
- In the toy the partition is built at once from all observations that are routed; an MPI code would use a parallel bisection.

**Setup.** Realistic SEVIRI layout (§2.9), ℓ = 60 km (provisional), α = 0.8, symmetrised, one patch per task. Script: `experiments/partitions.py`; raw output: `experiments/partitions.out`.
- Load imbalance is max/mean over **all** patches, since a patch without data is an idle task: n³ is the setup cost, n² the cost per application.
- The one-image runs use 5 300 pixels at the calibrated spacing; accuracy depends on the local density only (§2.9). The patch counts with data therefore differ from §2.9's 11 000-pixel image.

**One image** (`bands/obs` rows and h = 4ℓ in the `.out` file):

| patches | partition | h/ℓ | rel. L2 error | with data | max/mean core | max/mean n³ | largest local n | sum of local R | bytes / apply | messages / apply |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | bands/rect | 1 | 3.8e-3 | 17 | 7.7 | 19.1 | 903 | 23 MB | 90 kB | 1 538 |
| 48 | bands/rect | **2** | **4.5e-4** | 17 | 7.7 | 18.8 | **951** | 26 MB | 96 kB | 1 540 |
| 48 | bisection | 2 | 2.6e-3 | 48 | 1.0 | 2.0 | 194 | 9 MB | 116 kB | 4 444 |
| 48 | bisection | **3** | **2.3e-4** | 48 | 1.0 | 2.4 | **256** | 14 MB | 142 kB | 4 460 |
| 192 | bands/rect | 1 | 6.8e-3 | 46 | 17.5 | 52.8 | 535 | 13 MB | 98 kB | 6 526 |
| 192 | bands/rect | **2** | **6.4e-4** | 46 | 17.5 | 47.9 | **572** | 16 MB | 111 kB | 6 564 |
| 192 | bisection | 2 | 4.3e-3 | 192 | 1.0 | 2.0 | 68 | 4 MB | 163 kB | 11 646 |
| 192 | bisection | **3** | **4.2e-4** | 192 | 1.0 | 2.2 | **100** | 9 MB | 232 kB | 11 834 |

**Two discs** (Meteosat-11 + Meteosat-8, channel 5: 19 584 pixels in 2 groups, as in one CCMA slot), cost only, at the equal-accuracy pair (bands/rect 2ℓ, bisection 3ℓ). The same-halo rows (4ℓ) are in the `.out` file.

| patches | partition | h/ℓ | with data | max/mean core | max/mean n³ | largest local n | sum of local R | bytes / apply | messages / apply |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 48 | bands/rect | 2 | 27 | 5.4 | 10.2 | 2 481 | 152 MB | 355 kB | 2 410 |
| 48 | bisection | 3 | 48 | 1.0 | 2.1 | 691 | 95 MB | 450 kB | 4 984 |
| 192 | bands/rect | 2 | 77 | 7.1 | 14.3 | 940 | 76 MB | 413 kB | 18 160 |
| 192 | bisection | 3 | 192 | 1.0 | 3.3 | 292 | 49 MB | 630 kB | 32 456 |

- **Equal-area bands leave most patches idle.** For SEVIRI they are badly imbalanced: only 17 of 48 or 46 of 192 patches hold data in one image, the busiest patch has 8–18× the mean core, and 18–53× the mean setup cost.
- **Bisection balances the load almost perfectly:** cores are equal; setup cost max/mean is 2–3 (up to 4 with two discs) and application cost about 2. The remaining spread is the halo.
- **At the same halo width, bisection is 4–7× less accurate** than bands/rect (h = 2–4ℓ; 10× at 1ℓ).
  - A measured cause: its `obs` halo is narrower than the geometric `rect` halo. On the same band patches, `obs` alone costs 2–3× in error at 2–4ℓ.
  - A likely further cause, not isolated here: all bisection patches lie on the data, so many more patch boundaries cross observations.
- **One ℓ more halo restores the accuracy.** For a 1e-3 gate (ℓ = 60 km), bands need 2ℓ (1ℓ fails: 3.8e-3 and 6.8e-3) and bisection 3ℓ (2.3e-4 and 4.2e-4).
- **At equal accuracy, bisection is far cheaper on the critical path**, the busiest task:
  - one image: the largest local system is 256 against 951 (48 patches), about 51× less n³; and 100 against 572 (192 patches), about 190× less;
  - two discs: 691 against 2 481 (about 46×) and 292 against 940 (about 33×);
  - total local memory is also lower: 14 against 26 MB, 9 against 16 MB, 95 against 152 MB, 49 against 76 MB.
- **The price is communication.** Bytes per application are 1.3–2.1× higher and messages 1.8–2.9× higher, because every patch has neighbours with data. The values are small (8 bytes each), so the main risk is message latency at large task counts. The task-level halo map of §2.8 and neighbour collectives mitigate it.
- **Recommendation for the library (plan D11):** density-balanced patches, by recursive bisection with tie-aware cuts so that co-located records stay together. The partition is deterministic given the data, so the reproducible mode works. The halo should be about one ℓ wider than geometric cells would need.
  - With many groups in a window, build the partition once from all of them (or per time slot), so that the total work per patch is balanced. This prototype builds it per run.
  - Caveat: accuracy was measured on one mock SEVIRI layout with one group; the two-disc runs are cost only.

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
- **The operator is non-symmetric.** Addressed by the symmetrised variant ½(Ã + Ãᵀ) (§2.5). Overlapping-Schwarz-style weighting of the halo results remains an untested alternative.
- **Model realism:**
  - add channels, inter-channel correlation and time slots (slide 1 assumptions 1 and 2);
  - ~~add a spherical or periodic domain~~ (done: §2.7);
  - add clear-sky gaps (cloud masks) to the geostationary pattern.
- **Decomposition:** weighted or recursive-bisection partitions for clustered observations.
