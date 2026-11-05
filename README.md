# Local strengthening for submodular function minimization

This project compares cutting-plane and minimum-norm methods for minimizing a submodular set function. Local strengthening adds nonnegative hinge terms from adjacent swaps of a greedy ordering. The LP methods use these terms in a lower model of the Lovász extension. The minimum-norm QP methods use the corresponding zonotopes inside the base polytope.

| Method | Implementation |
| --- | --- |
| `kelley` | Kelley LP with greedy supporting planes |
| `path` | LP with all eligible adjacent hinges |
| `path_25` | LP with at most 25 eligible hinges per model |
| `gd_mnp` | Wolfe minimum-norm point method with NumPy least squares and active-set updates |
| `mn` | Gurobi minimum-norm QP over greedy vertices |
| `path_mn` | Gurobi minimum-norm QP with full local strengthening |
| `path_mn25` | Gurobi minimum-norm QP with at most 25 eligible hinges per model by default |

The capped methods rank positive adjacent second differences by size, breaking ties by earlier position in the ordering. The cap applies separately to each new model. Both variants still evaluate all adjacent probes.

## Setup

Create a Python 3.11 or 3.12 environment and install the dependencies.

```sh
python -m venv .venv
# macOS or Linux
source .venv/bin/activate
# Windows PowerShell uses .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

All seven methods are available with a working Gurobi installation and a license that supports the benchmark model sizes. HiGHS supports `kelley`, `path` and `path_25`. The NumPy implementation of `gd_mnp` also runs under the HiGHS runner setting.

## Tests

```sh
python test.py --results-only
python test.py --solver highs
python test.py --solver gurobi
```

The first command verifies the upstream archive, source and data hashes, benchmark coverage, and agreement of the saved CSV, metadata and JSON records. It requires no solver calls. The other commands run every published dataset and seed through the normal CLI on ten-variable restrictions, check exact optima and base inequalities by enumerating subsets, and exercise result aggregation. The HiGHS setting runs 200 smoke solves. The Gurobi setting runs 350 and includes the QP methods.

The solver modes also invoke `check.py` for formulation and certificate checks and `benchmark_tests.py` for runner and aggregation regressions. Gurobi mode additionally invokes `consolidation_tests.py` with its regression fixture. The full formulation checks require PyMaxflow. Use `--instances-only` to run the archive, instance and aggregation checks separately from these additional suites. Smoke outputs use a temporary directory by default. Pass `--out smoke_results` to keep them.

## Data

The unchanged upstream release is in `data/bach_original.zip`. See Francis Bach, *Learning with Submodular Functions: A Convex Optimization Perspective*, Section 12.1, and the author's software page for the original release and its terms. [Paper](https://arxiv.org/abs/1111.6453) · [Software and data](https://www.di.ens.fr/~fbach/submodular/)

| Dataset | Variables | Objective |
| --- | ---: | --- |
| Long | 573 | Symmetrized GENRMF source–sink cut with the source-only cut value subtracted |
| Wide | 430 | Symmetrized GENRMF source–sink cut with the source-only cut value subtracted |
| Speech | 800 | Sentence selection with objective `-len(A) + 22 sqrt(covered_words(A))` |
| Image | 2500 | Dense weighted pixel cut with label terms on a 50 × 50 image |
| Moons | 400 | Log-determinant mutual information with label terms on a noisy two-moons dataset |

Long and Wide use ten released graphs each. Speech seeds permute the same sentence incidence data. Image and Moons use Python-generated instances based on the published constructions. The saved Image results use `--image-mode corrected`, which keeps labels aligned with the permuted graph. See `data.py` for objective definitions and instance generation.

## Final benchmark

`results_150s/` contains the final 350 runs, covering five datasets, seeds 1–10 and seven methods with one repeat per combination. The recorded setup used the full instance dimensions, a 150-second limit per solve, up to 2000 iterations, and one solver and BLAS thread. The run files record 186 optimal statuses, 163 time limits and one iteration limit.

- `runs.csv` is the per-run table.
- `config.json` and `invocations.jsonl` record settings, software versions, hashes and run coverage.
- The 50 instance JSON files record the sizes and fingerprints.
- The 350 run JSON files contain final incumbents, certificates and per-iteration histories under `history`.

The final gap is the incumbent upper bound minus the certificate lower bound, floored at zero for reporting. Certification uses `U - L <= atol + rtol * max(1, abs(U), abs(L))`, with `atol=1e-7` and `rtol=1e-9`. A time or iteration limit preserves the available bounds. Elapsed seconds exclude instance preparation and include solver setup and cleanup. The time budget is cooperative, so native calls may run slightly past the deadline. The `queries` field counts distinct subsets stored in the oracle cache. The `requests` field also counts repeat requests.

Summarize the bundled results or reproduce the benchmark in a separate directory.

```sh
python aggregate_results.py --input results_150s/runs.csv --out summaries
python run.py --solver gurobi --seed-range 1 10 --seconds 150 --out results_reproduced
```

The runner checkpoints each completed run, resumes compatible configurations, and regenerates the run table from JSON on resume. Reproductions on different platforms or dependency versions can have different floating-point fingerprints and timings. Use a fresh output directory for a changed configuration. `python run.py --help` lists all options.
