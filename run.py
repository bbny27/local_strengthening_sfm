"""Run selected SFM methods and write complete, inspectable experiment results."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from data import ARCHIVE, DATASETS, load
from sfm import BASE_METHODS, METHODS, METRICS, solve


FIELDS = [
    "dataset",
    "seed",
    "repeat",
    "n",
    "method",
    "solver",
    "status",
    "upper",
    "lower",
    "gap",
    "seconds",
    "queries",
    "requests",
    "kernel_batches",
    "iterations",
    "models",
    "minor_steps",
    "oracle_seconds",
    "lp_seconds",
    "lp_rows",
    "lp_columns",
    "error",
]
FIELDS += [key for key in METRICS if key not in FIELDS]
FIELDS += ["cpu_seconds", "selected_probes", "probe_requests", "scalar_evaluations",
           "fw_gap", "qp_objective", "master_objective", "experiment_seed", "fingerprint", "config_id"]

def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if rows:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(clean(rows))
        temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
    )
    p.add_argument("--solver", choices=["highs", "gurobi"], default="gurobi")
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--seed-range", nargs=2, type=int, metavar=("FIRST", "LAST"))
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--retained-terms", type=int, default=25, help="path_mn25 cap; path_25 is always capped at 25")
    p.add_argument("--recovery-every", type=int, default=10)
    p.add_argument("--experiment-seed", type=int, default=None, help="Algorithm RNG seed; separate from dataset generation")
    p.add_argument(
        "--seconds",
        type=float,
        default=150,
        help="Per method/run, excluding dataset preparation",
    )
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--atol", type=float, default=1e-7)
    p.add_argument("--rtol", type=float, default=1e-9)
    p.add_argument(
        "--limit-n",
        type=int,
        default=0,
        help="Smoke-test restriction; omitted coordinates fixed to zero",
    )
    p.add_argument(
        "--image-size",
        type=int,
        default=50,
        help="50 reproduces Bach's size; smaller is a modified instance",
    )
    p.add_argument(
        "--image-mode", choices=["corrected", "released"], default="corrected"
    )
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()
    if args.seeds is not None and args.seed_range is not None:
        p.error("Choose --seeds or --seed-range")
    args.seeds = (list(range(args.seed_range[0], args.seed_range[1] + 1))
                  if args.seed_range else args.seeds if args.seeds is not None else list(range(1, 11)))
    args.seeds = list(dict.fromkeys(args.seeds))
    args.methods = list(dict.fromkeys(args.methods))
    args.datasets = list(dict.fromkeys(args.datasets))
    if not args.seeds or min(args.seeds) < 1:
        p.error("Use a nonempty positive seed range/list")
    args.probe_budget, args.aggregation_every, args.recent_models = 25, 20, 20
    if (
        not np.isfinite(args.seconds + args.atol + args.rtol)
        or args.seconds <= 0
        or args.repeats < 1
        or args.max_iter < 1
        or args.limit_n < 0
        or args.image_size < 2
        or args.atol < 0
        or args.rtol < 0
        or min(args.retained_terms, args.probe_budget, args.recovery_every,
               args.aggregation_every, args.recent_models) < 0
        or (args.experiment_seed is not None and args.experiment_seed < 0)
    ):
        p.error(
            "Use positive budgets/repeats/iterations, image-size >= 2, and nonnegative tolerances/limit-n."
        )
    if args.solver != "gurobi" and any(m not in BASE_METHODS for m in args.methods):
        p.error("Experimental methods require --solver gurobi; no solver substitution is performed.")
    if any(name in args.datasets for name in ["long", "wide"]) and any(
        not 1 <= s <= 10 for s in args.seeds
    ):
        p.error("GENRMF seeds must be between 1 and 10.")
    args.out.mkdir(parents=True, exist_ok=True)
    versions = {}
    for package in [
        "numpy",
        "scipy",
        "Pillow",
        "matplotlib",
        "threadpoolctl",
        "gurobipy",
        "PyMaxflow",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    config = dict(
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in {"out", "seeds", "seed_range", "methods", "datasets", "repeats"}
        },
        python=platform.python_version(),
        platform=platform.platform(),
        packages=versions,
        archive_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
        blas_threads=1,
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ["sfm.py", "data.py", "run.py", "check.py", "requirements.txt"]},
    )
    config_path = args.out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text(encoding="utf-8")) != config:
            p.error("Configuration differs: use a new --out directory; previous results preserved.")
    elif any(args.out.iterdir()):
        p.error("Nonempty output lacks compatible config.json; use a new --out directory.")
    else:
        write_json(config_path, config)
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    all_runs = []
    for path in args.out.glob("*_r*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("config_id") != config_id:
            p.error(f"Checkpoint configuration mismatch: {path}")
        all_runs.append(result)
    done = {(r["dataset"], r["seed"], r["repeat"], r["method"]) for r in all_runs}
    rows = [{key: r.get(key, "") for key in FIELDS} for r in all_runs]
    write_csv(args.out / "runs.csv", rows)
    with (args.out / "invocations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(datasets=args.datasets, methods=args.methods,
                                    seeds=args.seeds, repeats=args.repeats, config_id=config_id)) + "\n")
    with threadpool_limits(limits=1):
        for name in args.datasets:
            for seed in args.seeds:
                if all((name, seed, repeat, method) in done
                       for repeat in range(1, args.repeats + 1) for method in args.methods):
                    continue
                tick = perf_counter()
                f, preparation_error = None, None
                meta = {"fingerprint": None}
                try:
                    f = load(name, seed, args.image_mode, args.image_size).restrict(args.limit_n, seed)
                    meta = dict(f.meta, n=f.n, fingerprint=f.fingerprint(),
                                preparation_seconds=perf_counter() - tick)
                    write_json(args.out / f"{name}_seed{seed}_instance.json", meta)
                except Exception as error:
                    preparation_error = error
                stem = f"{name}_seed{seed}_n{f.n if f is not None else 'unknown'}"
                group = []
                for repeat in range(1, args.repeats + 1):
                    # Rotate order to reduce systematic first-run effects. Each method has a fresh cache.
                    order = (
                        args.methods[(repeat - 1) % len(args.methods) :]
                        + args.methods[: (repeat - 1) % len(args.methods)]
                    )
                    for method in order:
                        if (name, seed, repeat, method) in done:
                            continue
                        run_start = perf_counter()
                        tag = f"{stem}_{method}_r{repeat}"
                        try:
                            if preparation_error:
                                raise preparation_error
                            result = solve(
                                f,
                                method,
                                args.solver,
                                args.seconds,
                                args.max_iter,
                                args.atol,
                                args.rtol,
                                retained_terms=args.retained_terms,
                                probe_budget=args.probe_budget,
                                recovery_every=args.recovery_every,
                                aggregation_every=args.aggregation_every,
                                recent_models=args.recent_models,
                                experiment_seed=args.experiment_seed if args.experiment_seed is not None else seed,
                            )
                            result.update(
                                dataset=name,
                                seed=seed,
                                repeat=repeat,
                                fingerprint=meta["fingerprint"],
                                experiment_seed=args.experiment_seed if args.experiment_seed is not None else seed,
                            )
                        except Exception as error:
                            result = dict(
                                dataset=name,
                                seed=seed,
                                repeat=repeat,
                                n=f.n if f is not None else None,
                                seconds=perf_counter() - run_start,
                                solver=args.solver,
                                method=method,
                                status="error",
                                error=f"{type(error).__name__}: {error}",
                            )
                        result["experiment_seed"] = args.experiment_seed if args.experiment_seed is not None else seed
                        result["config_id"] = config_id
                        result["fingerprint"] = meta["fingerprint"]
                        if method in ("gd_mnp", "mn", "path_mn", "path_mn25"):
                            for key in ("lp_seconds", "lp_rows", "lp_columns", "lp_solves",
                                        "lp_setup_seconds", "lp_optimize_seconds", "active_lp_rows"):
                                if key in result:
                                    result[key] = None
                        if method not in ("mn", "path_mn", "path_mn25"):
                            for key in ("qp_seconds", "qp_rows", "qp_columns", "qp_solves", "qp_objective", "fw_gap"):
                                if key in result:
                                    result[key] = None
                        if method != "gd_mnp" and "minor_steps" in result:
                            result["minor_steps"] = None
                        write_json(args.out / f"{tag}.json", result)
                        if "history" in result:
                            write_csv(
                                args.out / f"{tag}_history.csv", result["history"]
                            )
                        all_runs.append(clean(result))
                        rows = [{key: r.get(key, "") for key in FIELDS} for r in all_runs]
                        write_csv(
                            args.out / "runs.csv", rows
                        )  # Checkpoint after every completed run.
                        group.append(result)
                        print(
                            f"{tag}: {result['status']}  U={result.get('upper', '?')}  L={result.get('lower', '?')}  time={result.get('seconds', 0):.3f}s",
                            flush=True,
                        )

    print(f"Results: {args.out.resolve()}")
    if any(r["status"] == "error" for r in all_runs):
        raise SystemExit("Some runs failed; inspect error fields in runs.csv.")


if __name__ == "__main__":
    main()
