"""Adjacent-path cutting planes, Kelley, and Wolfe's minimum-norm-point method."""

from time import perf_counter, process_time
import numpy as np
from scipy import sparse
from scipy.optimize import linprog


BASE_METHODS = ["path", "kelley", "gd_mnp", "path_25"]
NORM_METHODS = ["path_mn", "path_mn25", "mn"]
METHODS = BASE_METHODS + NORM_METHODS
METRICS = dict(
    outer_iterations=0, lp_solves=0, qp_solves=0, separation_passes=0,
    separation_seconds=0., aggregation_seconds=0., cut_seconds=0.,
    recovery_calls=0, recovery_improvements=0, qp_seconds=0.,
    lp_seconds=0., lp_setup_seconds=0., lp_optimize_seconds=0.,
    active_lp_rows=0, archived_path_models=0, retained_path_terms=0,
    active_pairs=0, qp_rows=0, qp_columns=0, lp_rows=0, lp_columns=0,
    max_archive_violation=0., archive_separated=False, compressions=0,
)


class NumericalStall(RuntimeError):
    """A repeated piece or ordering failed to close the certified SFM gap."""


def check_time(deadline):
    if perf_counter() >= deadline:
        raise TimeoutError("Total time budget exhausted")


def mixing_weights(weights):
    weights = np.maximum(0., np.asarray(weights, dtype=float))
    total = weights.sum()
    if not np.isfinite(total) or total == 0:
        raise RuntimeError("Optimizer returned no finite mixing weights")
    return weights / total


def probe_positions(n, budget, rng):
    """Select before observing deltas; this RNG is independent of the dataset."""
    count = min(n - 1, budget)
    return (np.arange(n - 1) if count == n - 1 else
            np.sort(rng.choice(n - 1, count, replace=False)))


def path_model(g, order, values, delta, positions=None, cap=None):
    """Validate observed differences and build the existing centered minorant."""
    positions = np.arange(len(delta)) if positions is None else np.asarray(positions, dtype=int)
    d = delta[positions]
    alternate = d + values[positions] + values[positions + 2] - values[positions + 1]
    scale = np.maximum(1., np.maximum(np.abs(alternate), np.maximum(
        np.abs(values[positions]), np.maximum(
            np.abs(values[positions + 1]), np.abs(values[positions + 2])))))
    if np.any(d < -1e-10 * scale):
        raise RuntimeError("Negative adjacent difference at an observed probe: "
                           "oracle is not submodular")
    d = np.where(d > 1e-12 * scale, np.maximum(d, 0.), 0.)
    if cap is not None:
        keep = np.argsort(-d, kind="stable")[:cap]
        positions, d = positions[keep], d[keep]
    a, edges = g.copy(), {}
    for k, amount in zip(positions, d):
        if amount > 0:
            i, j = map(int, order[k:k + 2])
            a[i] -= amount / 2
            a[j] += amount / 2
            edges[min(i, j), max(i, j)] = float(amount / 2)
    return a, edges


def model_value(model, x):
    a, edges = model
    return float(a @ x + sum(w * abs(x[i] - x[j]) for (i, j), w in edges.items()))


def supporting_piece(model, x, order):
    """At ties select the unactivated directed hinge in the generating order."""
    a, edges = model
    s = a.copy()
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    for (i, j), w in edges.items():
        sign = (1 if rank[i] < rank[j] else -1) if x[i] == x[j] else np.sign(x[i] - x[j])
        s[i] += sign * w
        s[j] -= sign * w
    return s


def aggregate(models, weights):
    weights = mixing_weights(weights)
    a = np.zeros_like(models[0][0])
    edges = {}
    for alpha, (b, pairs) in zip(weights, models):
        if alpha == 0:
            continue
        a += alpha * b
        for e, w in pairs.items():
            edges[e] = edges.get(e, 0.) + alpha * w
    return a, edges


def aggregate_cut(model, deadline=np.inf):
    """Float-capacity s-t cut; z=1 is the source side. No integer rounding."""
    check_time(deadline)
    import maxflow

    a, edges = model
    graph = maxflow.Graph[float](len(a), len(edges))
    nodes = graph.add_nodes(len(a))
    graph.add_grid_tedges(nodes, np.maximum(0., -a), np.maximum(0., a))
    for (i, j), w in edges.items():
        graph.add_edge(i, j, float(w), float(w))
    check_time(deadline)
    value = graph.maxflow() + np.minimum(a, 0.).sum()
    z = np.logical_not(graph.get_grid_segments(nodes)).astype(np.int8)
    return z, float(value)


def gurobi_model():
    """Use identical LP/QP threads and feasibility tolerances; clean up on failure."""
    import gurobipy as gp

    env = gp.Env(empty=True)
    model = None
    try:
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model(env=env)
        model.Params.Method = 1
        model.Params.Threads = 1
        model.Params.FeasibilityTol = 1e-9
        model.Params.OptimalityTol = 1e-9
        return env, model
    except BaseException:
        if model is not None:
            model.dispose()
        env.dispose()
        raise


class Oracle:
    """Each run has its own cache and incumbent; all queried subsets count equally."""

    def __init__(self, instance):
        self.instance = instance
        self.cache = {}
        self.requests = self.batches = 0
        self.probe_requests = self.selected_probes = self.scalar_evaluations = 0
        self.seconds = 0.0
        self.upper = np.inf
        self.z = np.zeros(instance.n, dtype=np.int8)

    def save(self, z, value):
        self.requests += 1
        key = np.packbits(z).tobytes()
        if key not in self.cache:
            self.cache[key] = float(value)
        value = self.cache[key]
        if value < self.upper:
            self.upper, self.z = value, z.copy()
        return value

    def chain(self, order):
        start = perf_counter()
        v, delta = self.instance.chain(order)
        self.batches += 1
        z = np.zeros(len(order), dtype=np.int8)
        v[0] = self.save(z, v[0])
        for k, i in enumerate(order):
            z[i] = 1
            v[k + 1] = self.save(z, v[k + 1])
        g = np.empty(len(order))
        g[order] = np.diff(v)
        self.seconds += perf_counter() - start
        return g, v, delta

    def evaluate(self, z):
        start = perf_counter()
        try:
            key = np.packbits(z).tobytes()
            if key in self.cache:
                value = self.cache[key]
            else:
                value = self.instance.value(z)
                self.scalar_evaluations += 1
            return self.save(z, value)
        finally:
            self.seconds += perf_counter() - start

    def probes(self, order, v, delta, stop, positions=None):
        start = perf_counter()
        positions = range(len(order) - 1) if positions is None else positions
        selected = set(map(int, positions))
        self.selected_probes += len(selected)
        z = np.zeros(len(order), dtype=np.int8)
        for k in range(len(order) - 1):
            i, j = order[k : k + 2]
            if k in selected:
                z[j] = 1
                self.save(z, delta[k] + v[k] + v[k + 2] - v[k + 1])
                self.probe_requests += 1
                if stop():
                    break
                z[j] = 0
            z[i] = 1
        self.seconds += perf_counter() - start


class GurobiMaster:
    """Keep one Gurobi environment and model throughout an algorithm run."""

    def __init__(self, n):
        import gurobipy as gp

        self.env, self.model = gurobi_model()
        self.x = list(self.model.addVars(n, lb=0., ub=1.).values())
        self.t = self.model.addVar(lb=-gp.GRB.INFINITY, obj=1.)
        self.u, self.links, self.rows = {}, {}, []

    def optimize(self, models, pairs, deadline, stats):
        import gurobipy as gp

        # Append only models not already present.
        for a, edges in models[len(self.rows):]:
            for i, j in edges:
                if (i, j) not in self.u:
                    u = self.model.addVar(lb=-gp.GRB.INFINITY)
                    self.u[i, j] = u
                    plus = self.model.addLConstr(
                        self.x[i] - self.x[j] - u, "<", 0.
                    )
                    minus = self.model.addLConstr(
                        self.x[j] - self.x[i] - u, "<", 0.
                    )
                    self.links[i, j] = (plus, minus)
            ids = np.flatnonzero(a)
            expr = gp.LinExpr(a[ids].tolist(), [self.x[i] for i in ids])
            expr += gp.LinExpr(list(edges.values()), [self.u[e] for e in edges])
            self.rows.append(self.model.addLConstr(expr - self.t, "<", 0.))

        self.model.update()
        stats.update(lp_rows=len(self.rows) + 2 * len(self.u),
                     lp_columns=len(self.x) + 1 + len(self.u),
                     active_lp_rows=len(self.rows) + 2 * len(self.u),
                     active_pairs=len(self.u))
        optimize_lp(self.model, deadline, stats)
        if self.model.Status == gp.GRB.TIME_LIMIT:
            raise TimeoutError("LP time limit")
        if self.model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi status {self.model.Status}")

        # Match the ordering expected by the existing certificate code.
        constraints = self.rows + [row for e in pairs for row in self.links[e]]
        point = np.array(self.model.getAttr("X", self.x))
        multipliers = -np.array(self.model.getAttr("Pi", constraints))
        return point, multipliers, self.model.ObjVal

    def close(self):
        self.model.dispose()
        self.env.dispose()


def optimize_lp(model, deadline, stats):
    """LP time includes setup and extraction; optimize time is nested within it."""
    check_time(deadline)
    model.Params.TimeLimit = deadline - perf_counter()
    tick = perf_counter()
    stats["lp_setup_seconds"] += tick - stats.pop("_lp_tick")
    stats["lp_solves"] += 1
    try:
        model.optimize()
    finally:
        stats["lp_optimize_seconds"] += perf_counter() - tick


def master(models, n, solver, remaining, session=None):
    """Solve the master LP, reusing a Gurobi session; return x and a base vector."""
    owned = session is None
    session = {} if owned else session
    stats = session.setdefault("stats", METRICS.copy())
    tick = perf_counter()
    stats["_lp_tick"] = tick
    try:
        return extended_master(models, n, solver, tick + remaining, session)
    finally:
        if "_lp_tick" in stats:
            stats["lp_setup_seconds"] += perf_counter() - stats.pop("_lp_tick")
        stats["lp_seconds"] += perf_counter() - tick
        if owned and "master" in session:
            session["master"].close()


def extended_master(models, n, solver, deadline, session):
    stats = session["stats"]
    pairs = sorted({e for _, edges in models for e in edges})
    index = {e: k for k, e in enumerate(pairs)}
    m, K = len(pairs), len(models)
    stats.update(lp_rows=K + 2 * m, lp_columns=n + 1 + m,
                 active_lp_rows=K + 2 * m, active_pairs=m)
    if solver == "gurobi":
        if "master" not in session:
            session["master"] = GurobiMaster(n)
        point, multipliers, objective = session["master"].optimize(
            models, pairs, deadline, stats)
    elif solver == "highs":
        point, multipliers, objective = highs_master(models, n, pairs, index, deadline, stats)
    else:
        raise ValueError(solver)
    alpha = mixing_weights(multipliers[:K])
    session["alpha"] = alpha
    a, edges = aggregate(models, alpha)
    capacity = np.array([edges.get(e, 0.) for e in pairs])
    q = np.clip(multipliers[K::2] - multipliers[K + 1::2], -capacity, capacity)
    for (i, j), amount in zip(pairs, q):
        a[i] += amount
        a[j] -= amount
    return point, a, dict(lp_rows=K + 2 * m, lp_columns=n + 1 + m,
                         master_objective=float(objective))


def highs_master(models, n, pairs, index, deadline, stats):
    m, K = len(pairs), len(models)
    row, col, val = [], [], []
    for r, (a, edges) in enumerate(models):
        ids = np.flatnonzero(a)
        row.extend([r] * len(ids))
        col.extend(ids)
        val.extend(a[ids])
        row.append(r)
        col.append(n)
        val.append(-1.0)
        for e, w in edges.items():
            row.append(r)
            col.append(n + 1 + index[e])
            val.append(w)
    for k, (i, j) in enumerate(pairs):
        row.extend([K + 2 * k] * 3 + [K + 2 * k + 1] * 3)
        col.extend([i, j, n + 1 + k] * 2)
        val.extend([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
    A = sparse.coo_matrix((val, (row, col)), shape=(K + 2 * m, n + 1 + m)).tocsr()
    c = np.zeros(A.shape[1])
    c[n] = 1.0
    bounds = [(0.0, 1.0)] * n + [(None, None)] * (m + 1)
    check_time(deadline)
    tick = perf_counter()
    stats["lp_setup_seconds"] += tick - stats.pop("_lp_tick")
    stats["lp_solves"] += 1
    try:
        result = linprog(
            c,
            A_ub=A,
            b_ub=np.zeros(A.shape[0]),
            bounds=bounds,
            method="highs-ds",
            options=dict(
                time_limit=max(0.001, deadline - perf_counter()),
                primal_feasibility_tolerance=1e-9,
                dual_feasibility_tolerance=1e-9,
            ),
        )
    finally:
        stats["lp_optimize_seconds"] += perf_counter() - tick
    if not result.success:
        raise (TimeoutError("LP time limit") if result.status == 1
               else RuntimeError(result.message))
    return result.x[:n], -result.ineqlin.marginals, result.fun


class RowMaster:
    """Persistent n+1-variable LP. Each stored slope carries its original model ID."""

    def __init__(self, n):
        self.env, self.model = gurobi_model()
        self.x = list(self.model.addVars(n, lb=0., ub=1.).values())
        self.t = self.model.addVar(lb=-np.inf, obj=1.)
        self.rows, self.slopes, self.origins, self.keys = [], [], [], set()
        self.initialized = 0

    def add(self, origin, slope):
        import gurobipy as gp

        key = origin, slope.tobytes()
        if key in self.keys:
            return False
        ids = np.flatnonzero(slope)
        expr = gp.LinExpr(slope[ids].tolist(), [self.x[i] for i in ids])
        self.rows.append(self.model.addLConstr(expr - self.t, "<", 0.))
        self.slopes.append(slope.copy())
        self.origins.append(origin)
        self.keys.add(key)
        return True

    def optimize(self, count, deadline, stats):
        import gurobipy as gp

        self.model.update()
        stats.update(lp_rows=len(self.rows), lp_columns=len(self.x) + 1,
                     active_lp_rows=len(self.rows), active_pairs=0)
        optimize_lp(self.model, deadline, stats)
        if self.model.Status != gp.GRB.OPTIMAL:
            raise (TimeoutError("Row LP time limit") if self.model.Status == gp.GRB.TIME_LIMIT
                   else RuntimeError(f"Gurobi row LP status {self.model.Status}"))
        weights = mixing_weights(-np.array(self.model.getAttr("Pi", self.rows)))
        y = sum(w * s for w, s in zip(weights, self.slopes))
        alpha = np.bincount(self.origins, weights=weights, minlength=count)
        return np.array(self.model.getAttr("X", self.x)), y, self.model.ObjVal, alpha

    def close(self):
        self.model.dispose()
        self.env.dispose()


def archive_violations(models, x, objective, deadline, stats, atol, rtol):
    tick = perf_counter()
    stats["separation_passes"] += 1
    stats["archive_separated"] = False
    stats["max_archive_violation"] = np.nan  # unknown if this scan is interrupted
    violated, residual = [], 0.
    try:
        for r, model in enumerate(models):
            check_time(deadline)
            value = model_value(model, x)
            violation = value - objective
            residual = max(residual, violation)
            tol = max(1e-9, .1 * (atol + rtol * max(1., abs(value), abs(objective))))
            if violation > tol:
                violated.append(r)
        check_time(deadline)
        stats["max_archive_violation"] = residual
        stats["archive_separated"] = not violated
        return violated
    finally:
        stats["separation_seconds"] += perf_counter() - tick


def rows_master(models, greedies, orders, n, session, deadline, accept, atol, rtol):
    stats = session["stats"]
    stats["archive_separated"] = False
    while True:
        tick = perf_counter()
        stats["_lp_tick"] = tick
        try:
            check_time(deadline)
            if "master" not in session:
                session["master"] = RowMaster(n)
            lp = session["master"]
            for r in range(lp.initialized, len(models)):
                check_time(deadline)
                lp.add(r, greedies[r])
            lp.initialized = len(models)
            x, y, objective, alpha = lp.optimize(len(models), deadline, stats)
            session["alpha"] = alpha
            stats["master_objective"] = float(objective)
            accept(y)  # valid even if the following archive scan runs out of time
        finally:
            if "_lp_tick" in stats:
                stats["lp_setup_seconds"] += perf_counter() - stats.pop("_lp_tick")
            stats["lp_seconds"] += perf_counter() - tick
        violated = archive_violations(models, x, objective, deadline, stats, atol, rtol)
        if not violated:
            return x, y
        tick = perf_counter()
        try:
            added = 0
            for r in violated:
                check_time(deadline)
                added += lp.add(r, supporting_piece(models[r], x, orders[r]))
            if not added:
                raise NumericalStall("Archive violation persists at repeated linear pieces")
        finally:
            elapsed = perf_counter() - tick
            stats["lp_setup_seconds"] += elapsed
            stats["lp_seconds"] += elapsed


def aggregate_master(models, n, session, deadline, accept, atol, rtol,
                     compress=False, recent=20):
    """Compress only between outer rounds; restore originals until fully separated."""
    stats = session["stats"]
    stats["archive_separated"] = False
    state = session.setdefault("active", dict(models=[], ids=[], count=0))
    for r in range(state["count"], len(models)):
        state["models"].append(models[r])
        state["ids"].append(r)
    state["count"] = len(models)
    if compress and "expanded" in state:
        tick = perf_counter()
        try:
            check_time(deadline)
            provenance = np.pad(state["expanded"], (0, len(models) - len(state["expanded"])))
            merged = aggregate(models, provenance)
            keep = list(range(max(0, len(models) - recent), len(models)))
            # The newest original must have an initial representation, even at recent=0.
            keep = sorted(set(keep + [len(models) - 1]))
            if "master" in session:
                session.pop("master").close()
            state.update(models=[merged] + [models[r] for r in keep],
                         ids=[None] + keep, provenance=provenance)
            stats["compressions"] += 1
        finally:
            stats["aggregation_seconds"] += perf_counter() - tick
    while True:
        check_time(deadline)
        x, y, info = master(state["models"], n, "gurobi", deadline - perf_counter(), session)
        stats.update(info)
        accept(y)
        tick = perf_counter()
        try:
            expanded = np.zeros(len(models))
            for weight, origin in zip(session["alpha"], state["ids"]):
                if origin is None:
                    provenance = state["provenance"]
                    expanded[:len(provenance)] += weight * provenance
                else:
                    expanded[origin] += weight
            state["expanded"] = mixing_weights(expanded)
            session["original_alpha"] = state["expanded"]
        finally:
            stats["aggregation_seconds"] += perf_counter() - tick
        violated = archive_violations(models, x, info["master_objective"],
                                      deadline, stats, atol, rtol)
        if not violated:
            return x, y
        active = set(state["ids"])
        restore = [r for r in violated if r not in active]
        if not restore:
            raise NumericalStall("Archive violation persists in active extended models")
        for r in restore:
            state["models"].append(models[r])
            state["ids"].append(r)


class NormMaster:
    """Incremental convex QP over a hull of zonotopes, with shared pair flows."""

    def __init__(self, n):
        import gurobipy as gp

        self.env, self.model = gurobi_model()
        self.y = list(self.model.addVars(n, lb=-gp.GRB.INFINITY).values())
        self.balance = [self.model.addLConstr(v, "=", 0.) for v in self.y]
        self.mixing = self.model.addLConstr(gp.LinExpr(), "=", 1.)
        self.model.setObjective(.5 * gp.quicksum(v * v for v in self.y))
        self.alpha, self.q, self.caps = [], {}, {}

    def append(self, models, deadline):
        import gurobipy as gp

        for a, edges in models[len(self.alpha):]:
            check_time(deadline)
            for i, j in edges:
                if (i, j) not in self.q:
                    q = self.model.addVar(lb=-gp.GRB.INFINITY,
                                         column=gp.Column([-1., 1.],
                                                          [self.balance[i], self.balance[j]]))
                    self.q[i, j] = q
                    self.caps[i, j] = (self.model.addLConstr(q, "<", 0.),
                                       self.model.addLConstr(-q, "<", 0.))
            ids = np.flatnonzero(a)
            rows = [self.mixing] + [self.balance[i] for i in ids]
            vals = [1.] + (-a[ids]).tolist()
            for e, w in edges.items():
                rows.extend(self.caps[e])
                vals.extend([-w, -w])
            self.alpha.append(self.model.addVar(lb=0., column=gp.Column(vals, rows)))

    def optimize(self, models, deadline, stats, accept):
        import gurobipy as gp

        self.append(models, deadline)
        self.model.update()
        stats.update(qp_rows=len(self.y) + 1 + 2 * len(self.q),
                     qp_columns=len(self.y) + len(self.alpha) + len(self.q),
                     active_pairs=len(self.q))
        check_time(deadline)
        self.model.Params.TimeLimit = deadline - perf_counter()
        stats["qp_solves"] += 1
        self.model.optimize()
        if self.model.SolCount:
            alpha = mixing_weights(self.model.getAttr("X", self.alpha))
            y, capacity = aggregate(models, alpha)
            # Normalize and clip before reconstructing: solver residuals cannot invalidate a base.
            for e, variable in self.q.items():
                i, j = e
                flow = np.clip(variable.X, -capacity.get(e, 0.), capacity.get(e, 0.))
                y[i] += flow
                y[j] -= flow
            stats["qp_objective"] = float(.5 * (y @ y))
            accept(y)
        if self.model.Status != gp.GRB.OPTIMAL:
            raise (TimeoutError("Minimum-norm QP time limit")
                   if self.model.Status == gp.GRB.TIME_LIMIT else
                   RuntimeError(f"Gurobi QP status {self.model.Status}"))
        check_time(deadline)
        return y, alpha

    def close(self):
        self.model.dispose()
        self.env.dispose()


def solve(
    instance,
    method="path",
    solver="highs",
    seconds=150.0,
    max_iter=2000,
    atol=1e-7,
    rtol=1e-9,
    *,
    retained_terms=25,
    probe_budget=25,
    recovery_every=10,
    aggregation_every=20,
    recent_models=20,
    experiment_seed=1,
):
    """All bounds refer to normalized f(empty)=0; limits never imply convergence."""
    start = perf_counter()
    cpu_start = process_time()
    deadline = start + seconds
    if method not in METHODS:
        raise ValueError(method)
    if method not in BASE_METHODS and solver != "gurobi":
        raise ValueError(f"{method} requires --solver gurobi")
    if min(retained_terms, probe_budget, recovery_every, aggregation_every, recent_models) < 0:
        raise ValueError("Experimental controls must be nonnegative")
    O = Oracle(instance)
    n = instance.n
    lower, certificate = -np.inf, np.zeros(n)
    history, models, seen = [], [], set()
    stats = METRICS.copy()
    session = dict(stats=stats)
    greedies, orders = [], []
    rng = np.random.default_rng(np.random.SeedSequence([experiment_seed, 14101]))
    minor_steps = 0
    dimensions = dict(lp_rows=0, lp_columns=0)
    status = "iteration_limit"
    it = completed = 0

    def tolerance():
        return atol + rtol * max(
            1.0, abs(O.upper), abs(lower) if np.isfinite(lower) else 1.0
        )

    def bound(y):
        nonlocal lower, certificate
        candidate = np.minimum(y, 0).sum()
        if candidate > lower:
            lower, certificate = float(candidate), y.copy()
        if lower > O.upper + 10 * tolerance():
            raise RuntimeError("Lower bound exceeds incumbent: check oracle/numerics")

    def done():
        return np.isfinite(lower) and O.upper - lower <= tolerance()

    def record(it):
        history.append(
            dict(
                iteration=it,
                seconds=perf_counter() - start,
                lower=lower,
                upper=O.upper,
                gap=max(0.0, O.upper - lower),
                queries=len(O.cache),
                requests=O.requests,
                oracle_seconds=O.seconds,
                cpu_seconds=process_time() - cpu_start,
                selected_probes=O.selected_probes,
                probe_requests=O.probe_requests,
                scalar_evaluations=O.scalar_evaluations,
                **{k: v for k, v in stats.items() if not k.startswith("_")},
            )
        )

    try:
        if method in ("path", "path_25", "kelley"):
            x = np.zeros(n)
            for it in range(1, max_iter + 1):
                stats["outer_iterations"] = it
                if perf_counter() - start >= seconds:
                    status = "time_limit"
                    break
                order = np.argsort(-x, kind="stable")
                positions = None
                g, values, delta = O.chain(order)
                if done():
                    status = "optimal"
                    record(it)
                    break
                if method != "kelley":
                    O.probes(order, values, delta, lambda: done() or perf_counter() >= deadline,
                             positions=positions)
                    if done():
                        status = "optimal"
                        record(it)
                        break
                key = order.tobytes()
                if key in seen:
                    status = "numerical_stall"
                    record(it)
                    break
                seen.add(key)
                model = ((g.copy(), {}) if method == "kelley" else
                         path_model(g, order, values, delta, positions,
                                    25 if method == "path_25" else None))
                models.append(model)
                greedies.append(g.copy())
                orders.append(order.copy())
                stats["archived_path_models"] = len(models) if method != "kelley" else 0
                stats["retained_path_terms"] += len(model[1])
                stats["archive_separated"] = False
                remaining = seconds - (perf_counter() - start)
                if remaining <= 0:
                    status = "time_limit"
                    record(it)
                    break
                x, y, dimensions = master(
                    models, n, solver, remaining, session
                )
                stats.update(dimensions)
                stats["archive_separated"] = True
                dimensions.update({k: stats[k] for k in ("lp_rows", "lp_columns")})
                bound(y)
                completed += 1
                record(it)
                if done():
                    status = "optimal"
                    break
        elif method in NORM_METHODS:
            order = np.arange(n)
            g, values, delta = O.chain(order)
            bound(g)
            for it in range(1, max_iter + 1):
                stats["outer_iterations"] = it
                if done():
                    status = "optimal"
                    break
                check_time(deadline)
                key = order.tobytes()
                if key in seen:
                    raise NumericalStall("Repeated minimum-norm pricing ordering")
                seen.add(key)
                if method == "mn":
                    model = g.copy(), {}
                else:
                    O.probes(order, values, delta,
                             lambda: done() or perf_counter() >= deadline)
                    if done():
                        status = "optimal"
                        break
                    check_time(deadline)
                    model = path_model(g, order, values, delta,
                                       cap=retained_terms if method == "path_mn25" else None)
                models.append(model)
                stats["archived_path_models"] = len(models) if method != "mn" else 0
                stats["retained_path_terms"] += len(model[1])
                tick = perf_counter()
                try:
                    if "master" not in session:
                        session["master"] = NormMaster(n)
                    y, alpha = session["master"].optimize(models, deadline, stats, bound)
                    session["alpha"] = alpha
                finally:
                    stats["qp_seconds"] += perf_counter() - tick
                # Threshold recovery and pricing share this complete chain with the next round.
                order = np.argsort(y, kind="stable")
                check_time(deadline)
                g, values, delta = O.chain(order)
                stats["fw_gap"] = float(y @ (y - g))
                completed += 1
                record(it)
                if done():
                    status = "optimal"
                    break
        elif method == "gd_mnp":
            g, _, _ = O.chain(np.arange(n))
            B, weights, x = g[:, None], np.ones(1), g.copy()
            bound(x)
            record(0)
            for it in range(1, max_iter + 1):
                stats["outer_iterations"] = it
                if done():
                    status = "optimal"
                    break
                if perf_counter() - start >= seconds:
                    status = "time_limit"
                    break
                s, _, _ = O.chain(np.argsort(x, kind="stable"))  # min x^T s
                bound(x)
                if done():
                    status = "optimal"
                    record(it)
                    break
                if x @ (x - s) <= 1e-12 * max(1.0, x @ x) or np.any(
                    np.linalg.norm(B - s[:, None], axis=0) <= 1e-12
                ):
                    status = "numerical_stall"
                    record(it)
                    break
                B, weights = np.column_stack((B, s)), np.r_[weights, 0.0]
                while True:  # Wolfe minor cycles, not a Frank-Wolfe line search.
                    if perf_counter() - start >= seconds:
                        raise TimeoutError("MNP time limit")
                    D = B[:, 1:] - B[:, [0]]
                    beta = np.linalg.lstsq(D, -B[:, 0], rcond=None)[0]
                    if perf_counter() - start >= seconds:
                        raise TimeoutError(
                            "MNP affine solve exhausted time budget"
                        )
                    affine = np.r_[1 - beta.sum(), beta]
                    minor_steps += 1
                    if affine.min() > 0:
                        weights = affine
                        break
                    negative = affine <= 0
                    denom = weights[negative] - affine[negative]
                    usable = denom > 0
                    if not usable.any():
                        raise RuntimeError("Degenerate MNP minor cycle")
                    theta = np.min(weights[negative][usable] / denom[usable])
                    weights += theta * (affine - weights)
                    keep = weights > 1e-12
                    B, weights = B[:, keep], weights[keep]
                    weights /= weights.sum()
                x = B @ weights
                bound(x)
                record(it)
            if done():
                status = "optimal"
        else:
            raise ValueError(method)
    except TimeoutError:
        status = "time_limit"
    except NumericalStall:
        status = "numerical_stall"
    finally:
        if "master" in session:
            session["master"].close()
    # Certification, never a repeated ordering or an optimization objective, establishes optimality.
    if done():
        status = "optimal"
    record(it)
    return dict(
        method=method,
        solver=solver if method != "gd_mnp" else "numpy_lstsq",
        n=n,
        status=status,
        upper=O.upper,
        lower=lower,
        gap=max(0.0, O.upper - lower),
        seconds=perf_counter() - start,
        queries=len(O.cache),
        requests=O.requests,
        kernel_batches=O.batches,
        oracle_seconds=O.seconds,
        iterations=history[-1]["iteration"],
        models=len(models),
        minor_steps=minor_steps,
        z=O.z.tolist(),
        certificate=certificate.tolist(),
        history=history,
        model_weights=(session.get("original_alpha", session.get("alpha", np.array([])))).tolist(),
        cpu_seconds=process_time() - cpu_start,
        selected_probes=O.selected_probes,
        probe_requests=O.probe_requests,
        scalar_evaluations=O.scalar_evaluations,
        **{k: v for k, v in stats.items() if not k.startswith("_")},
    )
