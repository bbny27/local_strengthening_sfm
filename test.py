"""Check the bundled benchmark and exercise the project on every published seed."""
import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from threadpoolctl import threadpool_limits

from data import ARCHIVE, DATASETS, load
from run import FIELDS
from sfm import BASE_METHODS, METHODS

ROOT = Path(__file__).resolve().parent
SEEDS = range(1, 11)
FINISHED_STATUSES = {"optimal", "time_limit", "iteration_limit"}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_key(result):
    return result["dataset"], int(result["seed"]), int(result["repeat"]), result["method"]


def check_records(directory, methods, *, restrictions=False):
    """Check coverage, provenance, checkpoint consistency and recorded bounds."""
    config = read_json(directory / "config.json")
    settings = config["arguments"]
    config_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    assert config["archive_sha256"] == digest(ARCHIVE), "Benchmark data archive hash differs"
    for name, expected in config["code_sha256"].items():
        assert digest(ROOT / name) == expected, f"Benchmark source hash differs for {name}"

    expected = {(dataset, seed, 1, method)
                for dataset, seed, method in product(DATASETS, SEEDS, methods)}
    invocations = [json.loads(line) for line in
                   (directory / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    invoked = set()
    for invocation in invocations:
        assert invocation["config_id"] == config_id, "Invocation configuration differs"
        invoked.update(product(invocation["datasets"], invocation["seeds"],
                               range(1, invocation["repeats"] + 1), invocation["methods"]))
    assert invoked == expected, "Invocation coverage differs"
    rows = read_csv(directory / "runs.csv")
    assert len(rows) == len(expected) and {run_key(row) for row in rows} == expected, (
        "Missing, duplicate or unexpected CSV runs"
    )
    instances = {}
    for dataset, seed in product(DATASETS, SEEDS):
        meta = read_json(directory / f"{dataset}_seed{seed}_instance.json")
        assert (meta["dataset"], meta["seed"]) == (dataset, seed), meta
        instances[dataset, seed] = meta
    assert len(list(directory.glob("*_instance.json"))) == len(instances), (
        "Unexpected instance metadata"
    )

    checkpoints = set()
    for row in rows:
        dataset, seed, repeat, method = run_key(row)
        meta = instances[dataset, seed]
        path = directory / f"{dataset}_seed{seed}_n{meta['n']}_{method}_r{repeat}.json"
        checkpoints.add(path)
        result = read_json(path)
        assert run_key(result) == (dataset, seed, repeat, method), path
        assert result["n"] == meta["n"] and result["fingerprint"] == meta["fingerprint"], path
        assert result["config_id"] == config_id, path
        for field in FIELDS:
            value = result.get(field)
            assert row[field] == ("" if value is None else str(value)), (path, field)
        assert result["status"] in FINISHED_STATUSES and not result.get("error"), path
        assert np.isfinite(result["seconds"]) and result["seconds"] >= 0, path
        assert len(result["z"]) == meta["n"] and set(result["z"]) <= {0, 1}, path
        certificate = np.asarray(result["certificate"])
        assert certificate.shape == (meta["n"],) and np.isfinite(certificate).all(), path
        upper, lower = result["upper"], result["lower"]
        assert np.isfinite([upper, lower, result["gap"]]).all(), path
        assert abs(np.minimum(certificate, 0).sum() - lower) < 2e-6, path
        tolerance = settings["atol"] + settings["rtol"] * max(1, abs(upper), abs(lower))
        assert lower - upper <= 10 * tolerance, path
        assert abs(result["gap"] - max(0, upper - lower)) < 2e-6, path
        assert result["history"], path
        last = result["history"][-1]
        for field in ("upper", "lower", "gap", "queries", "requests"):
            assert last[field] == result[field], (path, field)
        if restrictions or result["status"] == "optimal":
            assert result["status"] == "optimal" and upper - lower <= tolerance, path
    assert set(directory.glob("*_r*.json")) == checkpoints, "Unexpected run checkpoints"
    return rows


def check_restricted_instances(directory, rows):
    """Verify every small solve against enumeration, including its base certificate."""
    with threadpool_limits(limits=1):
        for dataset, seed in product(DATASETS, SEEDS):
            instance = load(dataset, seed, "corrected", 50).restrict(10, seed)
            subsets = np.asarray(list(product((0, 1), repeat=instance.n)), dtype=np.int8)
            values = np.asarray([instance.value(z) for z in subsets])
            optimum = values.min()
            fingerprint = instance.fingerprint()
            for row in rows:
                if (row["dataset"], int(row["seed"])) != (dataset, seed):
                    continue
                path = directory / f"{dataset}_seed{seed}_n{instance.n}_{row['method']}_r1.json"
                result = read_json(path)
                assert result["fingerprint"] == fingerprint, path
                assert abs(instance.value(result["z"]) - result["upper"]) < 2e-6, path
                assert result["lower"] <= optimum + 2e-6, path
                assert abs(result["upper"] - optimum) < 2e-6, path
                certificate = np.asarray(result["certificate"])
                assert np.max(subsets @ certificate - values) < 2e-6, path
                assert abs(certificate.sum() - values[-1]) < 2e-6, path


def command(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True)


def run_smoke(args, scratch):
    out = args.out.resolve() if args.out else scratch / "runs"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise SystemExit("Use a fresh or empty --out directory for smoke results.")
    methods = METHODS if args.solver == "gurobi" else BASE_METHODS
    command("run.py", "--solver", args.solver, "--methods", *methods,
            "--seed-range", 1, 10, "--seconds", 30, "--max-iter", 1000,
            "--limit-n", 10, "--out", out)
    rows = check_records(out, methods, restrictions=True)
    check_restricted_instances(out, rows)
    command("aggregate_results.py", "--input", out / "runs.csv", "--out", out / "summaries")
    assert len(read_csv(out / "summaries" / "paired_runs.csv")) == len(rows)
    assert len(read_csv(out / "summaries" / "dataset_method.csv")) == len(DATASETS) * len(methods)
    assert len(read_csv(out / "summaries" / "overall.csv")) == len(methods)
    print(f"PASS {len(rows)} restricted solves across all 50 instances and CSV aggregation",
          flush=True)
    if args.instances_only:
        print("Skipped the separate correctness and regression suites (--instances-only).")
        return
    command("check.py", "--solver", args.solver, "--out", scratch / "checks.json")
    command("benchmark_tests.py")
    if args.solver == "gurobi":
        command("consolidation_tests.py")
    else:
        print("Skipped Gurobi-only regression tests (--solver highs).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=("gurobi", "highs"), default="gurobi")
    parser.add_argument("--out", type=Path, help="Keep new smoke outputs in an empty directory")
    parser.add_argument("--instances-only", action="store_true",
                        help="Run the archive and CLI checks without the separate regression suites")
    parser.add_argument("--results-only", action="store_true",
                        help="Verify the shipped data and results without running any solver")
    args = parser.parse_args()
    if args.results_only and (args.out or args.instances_only):
        parser.error("--results-only cannot be combined with --out or --instances-only")
    with ZipFile(ARCHIVE) as archive:
        assert archive.testzip() is None, "Corrupt member in the dataset archive"
    final = ROOT / "results_150s"
    config = read_json(final / "config.json")["arguments"]
    assert config["solver"] == "gurobi" and config["seconds"] == 150
    assert config["limit_n"] == 0 and config["image_size"] == 50
    assert config["image_mode"] == "corrected"
    rows = check_records(final, METHODS)
    print(f"PASS bundled dataset archive and {len(rows)} final benchmark checkpoints", flush=True)
    if not args.results_only:
        with TemporaryDirectory(prefix="sfm_smoke_") as temporary:
            run_smoke(args, Path(temporary))


if __name__ == "__main__":
    main()
