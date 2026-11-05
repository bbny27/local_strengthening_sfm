"""Registry, original-result regressions, and instrumented top-25 selection."""
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits
from data import Instance
import sfm

EXPECTED_METHODS = {'path', 'kelley', 'gd_mnp', 'path_25', 'path_mn', 'path_mn25', 'mn'}


def main():
    assert set(sfm.METHODS) == EXPECTED_METHODS and len(sfm.METHODS) == 7
    cases = json.loads(Path(__file__).with_name('rename_baseline.json').read_text())
    checked = 0
    with threadpool_limits(limits=1):
        for case in cases:
            f = Instance('regression', 'cut', np.array(case['b']), np.array(case['C']))
            for old, new in [('mnp', 'gd_mnp'), ('vertex_mn', 'mn'), ('path', 'path')]:
                result = sfm.solve(f, new, 'gurobi', seconds=30)
                for key, expected in case['results'][old].items():
                    if key == 'status':
                        assert result[key] == expected
                    else:
                        np.testing.assert_allclose(result[key], expected, atol=2e-8, rtol=1e-9,
                                                   err_msg=f'{new}: {key}')
                checked += 1
            # <=25 eligible hinges: entire deterministic execution agrees.
            a, b = [sfm.solve(f, method, 'gurobi', seconds=30) for method in ('path', 'path_25')]
            for key in case['results']['path']:
                if key == 'status':
                    assert a[key] == b[key]
                else:
                    np.testing.assert_allclose(a[key], b[key], atol=2e-8, rtol=1e-9)
            for obsolete in ('mnp', 'vertex_mn', 'vertex_mnp', 'path_rows25', 'path_rows',
                             'path_rows_cut', 'path_probe25', 'path_agg_cut'):
                try:
                    sfm.solve(f, obsolete)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f'Obsolete method accepted: {obsolete}')
        rng = np.random.default_rng(991)
        n = 40
        C = np.triu(rng.random((n, n)), 1); C += C.T
        f = Instance('top25', 'cut', rng.normal(0, 20, n), C)
        native_model, native_master = sfm.path_model, sfm.master
        traces = {}
        for method in ('path', 'path_25'):
            calls, masters = [], []
            def capture(g, order, values, delta, positions=None, cap=None):
                result = native_model(g, order, values, delta, positions, cap)
                calls.append((g.copy(), order.copy(), values.copy(), delta.copy(), positions, cap, result))
                return result
            def capture_master(models, n, solver, remaining, session=None):
                masters.append((len(models), solver))
                return native_master(models, n, solver, remaining, session)
            with patch('sfm.path_model', capture), patch('sfm.master', capture_master):
                result = sfm.solve(f, method, 'gurobi', seconds=30, max_iter=4,
                                   retained_terms=1)  # path_25 cap must stay fixed at 25.
            assert calls and masters
            assert [m[0] for m in masters] == list(range(1, len(masters) + 1))
            for g, order, values, delta, positions, cap, model in calls:
                assert positions is None and cap == (25 if method == 'path_25' else None)
                assert np.count_nonzero(delta > 1e-10) > 25
                top = np.argsort(-delta, kind='stable')[:25]
                if method == 'path_25':
                    assert set(model[1]) == {tuple(sorted(map(int, order[k:k+2]))) for k in top}
                    expected = native_model(g, order, values, delta, positions=top)
                    np.testing.assert_allclose(model[0], expected[0])
                    assert model[1] == expected[1]
            traces[method] = calls
        # Initialization, full candidate generation and original oracle scores agree.
        for i in range(4):
            np.testing.assert_array_equal(traces['path'][0][i], traces['path_25'][0][i])
        # Equal scores choose the earlier adjacent position, including zero/ineligible hinges.
        g, order, values, delta, *_ = traces['path'][0]
        tied = native_model(g, order, values, np.ones(n - 1), cap=25)
        assert set(tied[1]) == {tuple(sorted(map(int, order[k:k+2]))) for k in range(25)}
        sparse = np.zeros(n - 1); sparse[:8] = 1
        full, capped = [native_model(g, order, values, sparse, cap=cap) for cap in (None, 25)]
        np.testing.assert_array_equal(full[0], capped[0]); assert full[1] == capped[1]
    print(f'PASS registry, {checked} original-result regressions, shared initialization, top-25 and row accumulation')


if __name__ == '__main__':
    main()
