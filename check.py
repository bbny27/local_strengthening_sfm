import argparse
from itertools import product
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from data import DATASETS, Instance, load
from scipy import sparse
from scipy.optimize import linprog
from time import perf_counter
from unittest.mock import patch
import sfm
from sfm import (BASE_METHODS, METHODS, METRICS, NormMaster, Oracle,
                 aggregate, aggregate_cut, aggregate_master, master, model_value,
                 path_model, probe_positions, rows_master, solve, supporting_piece)


def verify(f, solver, label, methods):
    Z = np.array(list(product([0, 1], repeat=f.n)), dtype=np.int8)
    values = np.array([f.value(z) for z in Z])
    optimum = values.min()
    rng = np.random.default_rng(17)
    for _ in range(3):
        order = rng.permutation(f.n)
        v, delta = f.chain(order)
        z = np.zeros(f.n, dtype=np.int8)
        assert abs(v[0] - f.value(z)) < 1e-7
        for k, i in enumerate(order):
            if k + 1 < f.n:
                q = z.copy()
                q[order[k + 1]] = 1
                assert abs(f.value(q) - (delta[k] + v[k] + v[k + 2] - v[k + 1])) < 1e-7
            z[i] = 1
            assert abs(v[k + 1] - f.value(z)) < 1e-7
        g = np.empty(f.n)
        g[order] = np.diff(v)
        H = Z @ g + np.maximum(Z[:, order[1:]] - Z[:, order[:-1]], 0) @ delta
        assert np.max(H - values) < 1e-7, "Invalid path minorant"
        x = rng.random(f.n)
        permutation = np.argsort(-x, kind="stable")
        prefix, _ = f.chain(permutation)
        lovasz = np.diff(prefix) @ x[permutation]
        plane = g @ x
        path = plane + delta @ np.maximum(x[order[1:]] - x[order[:-1]], 0)
        assert plane <= path + 1e-7 and path <= lovasz + 1e-7
    results = []
    for method in methods:
        r = solve(f, method, solver=solver, seconds=30, max_iter=1000,
                  aggregation_every=2, recent_models=1)
        assert r["status"] == "optimal", (label, method, r["status"], r["gap"])
        assert abs(r["upper"] - optimum) < 2e-6, (label, method, r["upper"], optimum)
        assert abs(f.value(r["z"]) - r["upper"]) < 1e-7
        y = np.array(r["certificate"])
        assert np.max(Z @ y - values) < 2e-6, "Invalid base certificate"
        assert abs(y.sum() - f.value(np.ones(f.n))) < 2e-6
        assert r["lower"] <= optimum + 2e-6
        results.append(
            dict(
                case=label,
                method=method,
                optimum=float(optimum),
                status=r["status"],
                gap=r["gap"],
                iterations=r["iterations"],
                queries=r["queries"],
            )
        )
    print(f"PASS {label}: optimum {optimum:.9g}", flush=True)
    return results


def assert_certificate(f, result):
    Z = np.array(list(product([0, 1], repeat=f.n)), dtype=np.int8)
    values = np.array([f.value(z) for z in Z])
    y = np.array(result["certificate"])
    assert np.isfinite(result["lower"])
    assert np.max(Z @ y - values) < 2e-6
    assert abs(y.sum() - values[-1]) < 2e-6
    assert abs(np.minimum(y, 0).sum() - result["lower"]) < 2e-6
    assert result["lower"] <= values.min() + 2e-6
    assert abs(f.value(result["z"]) - result["upper"]) < 2e-6


def corners(model):
    a, edges = model
    result = []
    for signs in product([-1., 1.], repeat=len(edges)):
        s = a.copy()
        for sign, ((i, j), w) in zip(signs, edges.items()):
            s[i] += sign * w
            s[j] -= sign * w
        result.append(s)
    return result


def archive_tests(toy, solver):
    """Independent corner LP/QP references; no oracle calls in archive separation."""
    rng = np.random.default_rng(723)
    comparisons = 0
    for trial in range(4):
        orders = [np.arange(toy.n)] + [rng.permutation(toy.n) for _ in range(3)]
        models, greedies = [], []
        for order in orders:
            values, delta = toy.chain(order)
            g = np.empty(toy.n)
            g[order] = np.diff(values)
            models.append(path_model(g, order, values, delta))
            greedies.append(g)
        n = toy.n
        Z = np.array(list(product([0, 1], repeat=n)))
        values = np.array([toy.value(z) for z in Z])
        pieces = np.array([s for model in models for s in corners(model)])
        # Explicit corner-expanded LP uses an independent SciPy formulation.
        result = linprog(np.r_[np.zeros(n), 1.],
                         A_ub=np.column_stack((pieces, -np.ones(len(pieces)))),
                         b_ub=np.zeros(len(pieces)), bounds=[(0., 1.)] * n + [(None, None)],
                         method="highs-ds")
        assert result.success
        session = {}
        try:
            _, _, info = master(models, n, solver, 20, session)
            assert abs(result.fun - info["master_objective"]) < 2e-7
            merged = aggregate(models, session["alpha"])
            z, cut = aggregate_cut(merged)
            binary = np.array([model_value(merged, z) for z in Z])
            assert abs(cut - binary.min()) < 2e-7
            assert abs(cut - info["master_objective"]) < 2e-7
            assert abs(cut - model_value(merged, z)) < 2e-7
            assert np.max(binary - values) < 2e-7
            comparisons += 1
        finally:
            if "master" in session:
                session["master"].close()
        if solver != "gurobi":
            continue
        for backend in ("rows", "aggregate"):
            session = dict(stats=METRICS.copy())
            certificates = []
            try:
                if backend == "rows":
                    rows_master(models, greedies, orders, n, session,
                                perf_counter() + 20, certificates.append, 1e-7, 1e-9)
                    assert session["stats"]["lp_columns"] == n + 1
                    alpha = session["alpha"]
                else:
                    aggregate_master(models[:2], n, session, perf_counter() + 20,
                                     certificates.append, 1e-7, 1e-9)
                    aggregate_master(models, n, session, perf_counter() + 20,
                                     certificates.append, 1e-7, 1e-9,
                                     compress=True, recent=0)
                    # Compress again to exercise expansion of previous aggregate provenance.
                    aggregate_master(models, n, session, perf_counter() + 20,
                                     certificates.append, 1e-7, 1e-9,
                                     compress=True, recent=0)
                    alpha = session["original_alpha"]
                    assert len(alpha) == len(models) and abs(alpha.sum() - 1) < 1e-10
                stats = session["stats"]
                assert stats["archive_separated"]
                assert stats["max_archive_violation"] < 2e-7
                assert abs(stats["master_objective"] - result.fun) < 2e-7
                _, cut = aggregate_cut(aggregate(models, alpha))
                assert abs(cut - result.fun) < 2e-7
                for y in certificates:
                    assert np.max(Z @ y - values) < 2e-6
                    assert abs(y.sum() - values[-1]) < 2e-6
                comparisons += 1
            finally:
                if "master" in session:
                    session["master"].close()
        qp_tests(toy, models, greedies, pieces, Z, values)
    # Unary signs, zero weights, and fractional capacities with no rounding.
    for a in (np.array([-1.3, 0., 2.7]), np.zeros(3), np.array([2., -3.2, -1.])):
        model = a, {(0, 1): .137, (1, 2): .29}
        Z = np.array(list(product([0, 1], repeat=3)))
        z, value = aggregate_cut(model)
        assert abs(value - min(model_value(model, v) for v in Z)) < 1e-10
        assert abs(value - model_value(model, z)) < 1e-10
    # Cut model values must never become incumbents; cached scalar calls are skipped.
    order = np.arange(4)
    v, d = toy.chain(order)
    g = np.diff(v)
    model = path_model(g, order, v, d)
    z, h = aggregate_cut(model)
    oracle = Oracle(toy)
    original = oracle.evaluate(z)
    assert original == toy.value(z) and original > h + 1e-7
    oracle.evaluate(z)
    assert oracle.scalar_evaluations == 1 and oracle.requests == 2
    assert oracle.upper == original and oracle.seconds > 0
    print(f"PASS {comparisons} frozen-archive comparisons, aggregate cuts and scalar cache", flush=True)


def qp_tests(f, models, greedies, pieces, Z, values):
    import gurobipy as gp

    qp = NormMaster(f.n)
    stats = METRICS.copy()
    try:
        # Append incrementally and compare to an explicit corner-mixture QP.
        qp.optimize(models[:1], perf_counter() + 20, stats, lambda y: None)
        y, alpha = qp.optimize(models, perf_counter() + 20, stats, lambda y: None)
        assert np.max(Z @ y - values) < 2e-6
        assert abs(y.sum() - values[-1]) < 2e-6
        assert stats["lp_seconds"] == 0 and stats["lp_solves"] == 0
        env, ref = sfm.gurobi_model()
        try:
            weights = list(ref.addVars(len(pieces), lb=0.).values())
            ref.addConstr(gp.quicksum(weights) == 1.)
            point = [gp.LinExpr(pieces[:, i].tolist(), weights) for i in range(f.n)]
            ref.setObjective(.5 * gp.quicksum(v * v for v in point))
            ref.optimize()
            assert ref.Status == gp.GRB.OPTIMAL
            assert abs(.5 * (y @ y) - ref.ObjVal) < 2e-6
        finally:
            ref.dispose()
            env.dispose()
    finally:
        qp.close()
    vertex = NormMaster(f.n)
    try:
        s, _ = vertex.optimize([(g, {}) for g in greedies], perf_counter() + 20,
                               METRICS.copy(), lambda y: None)
        assert y @ y <= s @ s + 2e-6
    finally:
        vertex.close()


def probe_tests():
    n = 32
    rng = np.random.default_rng(71)
    C = rng.random((n, n)); C = np.triu(C, 1); C += C.T
    f = Instance("probe tests", "cut", rng.normal(size=n), C)
    order = rng.permutation(n)
    O = Oracle(f)
    g, v, d = O.chain(order)
    selected = probe_positions(n, 25, np.random.default_rng(82))
    assert np.array_equal(selected, probe_positions(n, 25, np.random.default_rng(82)))
    assert len(selected) == 25
    before = set(O.cache)
    O.probes(order, v, d, lambda: False, positions=selected)
    assert O.probe_requests == 25 and len(O.cache) == n + 1 + 25
    observed = []
    for k in range(n - 1):
        z = np.zeros(n, dtype=np.int8); z[order[:k]] = 1; z[order[k + 1]] = 1
        key = np.packbits(z).tobytes()
        assert (key in O.cache) == (k in selected)
        if k in selected:
            observed.append(f.value(z))
    assert abs(O.upper - min(v.min(), min(observed))) < 1e-7
    model = path_model(g, order, v, d, cap=25)
    assert len(model[1]) == 25
    top = np.argsort(-d, kind="stable")[:25]
    assert set(model[1]) == {tuple(sorted(map(int, order[k:k + 2]))) for k in top}
    for kwargs in (dict(cap=0), dict(positions=[])):
        a, edges = path_model(g, order, v, d, **kwargs)
        assert not edges and np.allclose(a, g)
    full = path_model(g, order, v, d)
    selected_full = probe_positions(n, n, np.random.default_rng(1))
    a, edges = path_model(g, order, v, d, positions=selected_full)
    assert np.allclose(a, full[0]) and edges == full[1]
    O = Oracle(f); g, v, d = O.chain(order)
    O.probes(order, v, d, lambda: False, positions=[])
    assert O.probe_requests == 0 and len(O.cache) == n + 1
    # Unobserved deltas may not affect restricted construction or validity checks.
    altered = d.copy(); altered[np.setdiff1d(np.arange(n - 1), selected)] = -1e20
    a, edges = path_model(g, order, v, altered, positions=selected)
    original = path_model(g, order, v, d, positions=selected)
    assert np.array_equal(a, original[0]) and edges == original[1]
    # Exact delta ties retain earlier adjacent positions.
    tied = path_model(g, order, v, np.ones(n - 1), cap=25)
    assert set(tied[1]) == {tuple(sorted(map(int, order[k:k + 2]))) for k in range(25)}
    tiny = np.full(n - 1, -1e-13)
    assert not path_model(g, order, v, tiny)[1]
    try:
        path_model(g, order, v, -np.ones(n - 1))
        raise AssertionError("Negative deltas were accepted")
    except RuntimeError as error:
        assert "submodular" in str(error)
    # At a coordinate tie the generated piece is the original greedy vector.
    assert np.allclose(supporting_piece(full, np.zeros(n), order), g)
    assert before.issubset(O.cache)
    print("PASS active 25-term cap, restricted probes, zero budgets and deterministic ties", flush=True)


def limit_tests(toy, solver, methods):
    for method in methods:
        r = solve(toy, method, solver, max_iter=1, seconds=20, recovery_every=0)
        assert r["status"] in ("optimal", "iteration_limit")
        assert_certificate(toy, r)
    for method in ("path", "path_25"):
        g = np.diff(toy.chain(np.arange(toy.n))[0])
        with patch("sfm.master", return_value=(np.zeros(toy.n), g, {"lp_rows": 1, "lp_columns": toy.n + 1})):
            r = solve(toy, method, solver, seconds=20)
            assert r["status"] == "numerical_stall"
            assert_certificate(toy, r)
    if solver != "gurobi":
        return
    import gurobipy as gp
    # Exercise QP-optimization interruption and preserve the initial greedy certificate.
    with patch.object(gp.Model, "optimize", side_effect=TimeoutError("test native QP deadline")):
        r = solve(toy, "path_mn", solver, seconds=20)
        assert r["status"] == "time_limit" and r["qp_solves"] == 1
        assert r["lp_seconds"] == 0 and r["qp_seconds"] > 0
        assert_certificate(toy, r)
    # A returned QP time-limit solution is reconstructed before reporting the interruption.
    native = gp.Model.optimize
    def limited(model):
        model.Params.TimeLimit = 0.
        native(model)
    with patch.object(gp.Model, "optimize", limited):
        r = solve(toy, "path_mn", solver, seconds=20)
        assert r["status"] == "time_limit" and r["qp_solves"] == 1
        assert_certificate(toy, r)
    # Base greedy-point chain is queried exactly once and the pricing chain is reused.
    r = solve(toy, "mn", solver, max_iter=1, seconds=20)
    assert r["kernel_batches"] == r["qp_solves"] + 1 and r["probe_requests"] == 0
    print("PASS iteration/time limits, repeated orderings and interrupted certificates", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", choices=["highs", "gurobi"], default="highs")
    parser.add_argument("--out", type=Path, default=Path("checks.json"))
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        help="Defaults to all methods with Gurobi, four LP/NumPy methods with HiGHS")
    args = parser.parse_args()
    methods = args.methods or (METHODS if args.solver == "gurobi" else BASE_METHODS)
    if args.solver != "gurobi" and any(m not in BASE_METHODS for m in methods):
        parser.error("New methods require --solver gurobi")
    rows = []
    with threadpool_limits(limits=1):
        C = np.ones((4, 4)) - np.eye(4)
        toy = Instance("notes", "cut", np.array([4.0, -4.0, 3.0, -3.0]), C)
        # Section 7: the first path master is -5; the optimum is -3.
        v, delta = toy.chain(np.arange(4))
        assert np.allclose(v, [0, 7, 4, 6, 0])
        a = np.diff(v)
        a[:-1] -= delta / 2
        a[1:] += delta / 2
        _, y, info = master(
            [(a, {(i, i + 1): delta[i] / 2 for i in range(3)})], 4, args.solver, 20
        )
        assert abs(info["master_objective"] + 5) < 1e-8
        rows += verify(toy, args.solver, "notes four-variable example", methods)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            C = rng.random((8, 8))
            C = np.triu(C, 1)
            C += C.T
            b = rng.normal(0, 3, 8)
            rows += verify(
                Instance("random", "cut", b, C), args.solver, f"random cut {seed}", methods
            )
        rng = np.random.default_rng(8)
        W = sparse.csr_matrix(rng.random((8, 12)) < 0.3)
        rows += verify(
            Instance("coverage", "speech", rng.normal(-15, 10, 8), W),
            args.solver,
            "nontrivial coverage",
            methods,
        )
        A = rng.standard_normal((8, 8))
        K = A @ A.T + np.eye(8)
        rows += verify(
            Instance("logdet", "logdet", rng.normal(0, 3, 8), K, np.linalg.inv(K)),
            args.solver,
            "nontrivial logdet",
            methods,
        )
        for name in DATASETS:
            f = load(name, 1).restrict(10, 1)
            assert abs(f.value(np.zeros(f.n))) < 1e-10
            rows += verify(f, args.solver, f"Bach {name} restriction n=10", methods)
        for b in (np.array([-2.]), np.array([0.]), np.array([2.]),
                  np.array([-3., 0., 2., -1.])):
            f = Instance("modular", "cut", b, np.zeros((len(b), len(b))))
            rows += verify(f, args.solver, f"modular {b.tolist()}", methods)
        archive_tests(toy, args.solver)
        probe_tests()
        limit_tests(toy, args.solver, methods)
    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"All {len(rows)} algorithm checks passed; saved {args.out}")


if __name__ == "__main__":
    main()
