# r_matrix

This is a prototype for applying a **correlated observation-error covariance R⁻¹** to observations spread over many MPI tasks, as in an operational NWP system.

Observations start on mock MPI tasks in random order. The prototype:
1. redistributes them to spatial subdomains;
2. builds halos of nearby observations;
3. solves a local core + halo system on each task;
4. returns the results to the original tasks.

Each result is validated against an exact global solve. It runs in a single Python process; the MPI communication is simulated and its traffic is counted.

## Quick start

```bash
git clone https://github.com/marcin85PL/r_matrix.git
cd r_matrix
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all]"
python -m pytest -q                                   # 26 tests, a few seconds
python -m ipykernel install --user --name r_matrix   # optional: Jupyter kernel for the notebooks
```

```python
from r_matrix import Config, run
res = run(Config(), h=2.0)       # 3000 obs on 12 mock tasks, halo h = 4 correlation length scales
print(res.metrics["rel_l2"])     # relative error against the exact global solve
```

## Contents

| Path | What it is |
|---|---|
| [r_matrix/README.md](r_matrix/README.md) | **Start here.** Guide: setup, background, notation, design, how to extend, starter exercises |
| [r_matrix/RESULTS.md](r_matrix/RESULTS.md) | Findings to date: accuracy against halo width, sensitivities, robustness, costs |
| [notebooks/01_walkthrough.ipynb](notebooks/01_walkthrough.ipynb) | The whole pipeline by hand on 4 tasks and 50 observations |
| [notebooks/02_experiments.ipynb](notebooks/02_experiments.ipynb) | The experiments behind RESULTS.md |
| `r_matrix/` | The package |
| `tests/` | pytest suite |

## License

Apache License 2.0; see [LICENSE](LICENSE).
