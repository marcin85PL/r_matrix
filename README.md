# r_matrix

This is a prototype for applying a **correlated observation-error covariance R⁻¹** to observations spread over many MPI tasks, as in an operational NWP system.

Observations start on mock MPI tasks in random order. The prototype:
1. redistributes them to spatial subdomains;
2. builds halos of nearby observations;
3. applies an approximate R⁻¹ on each task;
4. returns the results to the original tasks.

It has two ways to approximate R⁻¹:

| method | idea | gives | code |
|---|---|---|---|
| **core + halo** | solve R_loc over each patch plus a halo on every side, keep the core rows, symmetrise | R⁻¹ | `pipeline.py`, `covariance.py` |
| **Vecchia factor** | a sparse inverse-Cholesky factor W, R̃⁻¹ = WᵀW: per datum (point) or per patch with a one-sided halo (block) | R⁻¹, R^−½ = W, R^½ z = W⁻¹z; SPD by construction | `vecchia.py`, `vecchia_eval.py` |

The block Vecchia factor is the candidate alternative to core + halo: about its accuracy at the same halo width, with half the halo. The point variant needs almost all-to-all communication and is rejected. The findings are in [VECCHIA_REPORT.md](VECCHIA_REPORT.md).

Each result is validated against an exact global solve. It runs in a single Python process; the MPI communication is simulated and its traffic is counted.

## Quick start

```bash
git clone https://github.com/marcin85PL/r_matrix.git
cd r_matrix
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m pytest -q                                   # 131 tests, about 1 min
python -m ipykernel install --user --name r_matrix   # optional: Jupyter kernel for the notebooks
```

Core + halo:

```python
from r_matrix import Config, run
res = run(Config(), h=2.0)       # 3000 obs on 12 mock tasks, halo h = 4 correlation length scales
print(res.metrics["rel_l2"])     # relative error against the exact global solve
```

Block Vecchia on a golden case:

```python
from r_matrix import vecchia as V, vecchia_eval as E
from r_matrix.decomposition import BisectionDecomposition
obs, params, hs, _ = E.golden_data("sph_geo_disc")       # 900 SEVIRI-like data on the sphere, one group
dec = BisectionDecomposition.from_points(obs["x"], obs["y"], obs["global_id"], 8)
b = E.build(obs, dec, "block", params, seq="colour", h=hs[0])   # distributed setup on 8 mock tasks
lam = V.gen_eigs_W(b.W().toarray(), E.exact_C(obs, *params[0]))
print(abs(lam - 1).max())        # about 3e-2: generalised eigenvalues of W^T W against the exact C^-1
```

## Contents

| Path | What it is |
|---|---|
| [r_matrix/README.md](r_matrix/README.md) | **Start here.** Guide: setup, background, notation, design, how to extend, starter exercises |
| [r_matrix/RESULTS.md](r_matrix/RESULTS.md) | Findings for core + halo: accuracy against halo width, sensitivities, robustness, costs |
| [VECCHIA_REPORT.md](VECCHIA_REPORT.md) | Findings for the Vecchia factor against core + halo: accuracy, setup and apply cost, communication, sampling, scaling to 1024 tasks |
| [notebooks/01_walkthrough.ipynb](notebooks/01_walkthrough.ipynb) | The core + halo pipeline by hand on 4 tasks and 50 observations |
| [notebooks/02_experiments.ipynb](notebooks/02_experiments.ipynb) | The experiments behind RESULTS.md |
| [notebooks/03_vecchia_prototype.ipynb](notebooks/03_vecchia_prototype.ipynb) | The Vecchia factor step by step (point and block) on the geo-disc golden case |
| [notebooks/04_vecchia_evaluation.ipynb](notebooks/04_vecchia_evaluation.ipynb) | The results behind VECCHIA_REPORT.md; reads `experiments/vecchia_results/` |
| `r_matrix/` | The package |
| `experiments/` | Scripts behind RESULTS.md and VECCHIA_REPORT.md, and the Vecchia results (JSON) |
| `tests/` | pytest suite |

## Golden data for a compiled implementation

`r_matrix/export.py` writes small deterministic cases (observations, parameters, and the outputs of
every pipeline stage) as raw little-endian `.bin` arrays plus a `manifest.json` that records the
toy commit. The C++ library tests (`2026-r-package`) read them. The cases cover core + halo only;
golden data for a block Vecchia stage would come from `vecchia.py`.

```bash
python -m r_matrix.export --list
python -m r_matrix.export all <outdir>        # or one case name; refuses a dirty toy unless --allow-dirty
python -m pytest tests/test_export.py         # round-trip and consistency tests
```

## License

Apache License 2.0; see [LICENSE](LICENSE).
