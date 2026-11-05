"""Small runner/resume and synthetic aggregation regressions; no full experiments."""
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
from data import Instance
import aggregate_results
import run


def invoke(function, arguments):
    with patch.object(sys, 'argv', ['test', *map(str, arguments)]), contextlib.redirect_stdout(io.StringIO()):
        function()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def rejected(function, arguments):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            invoke(function, arguments)
        except SystemExit as error:
            assert error.code
        else:
            raise AssertionError('Expected rejection')


def main():
    with tempfile.TemporaryDirectory(prefix='benchmark_test_', dir=Path(__file__).parent) as temporary:
        root = Path(temporary)
        out = root / 'resume'
        f = Instance('tiny', 'cut', np.array([4., -4., 3., -3.]), np.ones((4, 4)) - np.eye(4), meta={})
        def load(*args):
            return f
        base = ['--out', out, '--datasets', 'speech', '--methods', 'path', '--solver', 'highs', '--seconds', '1']
        with patch('run.load', load):
            invoke(run.main, [*base, '--seeds', '1'])
            original = next(out.glob('*_r1.json')).read_bytes()
            invoke(run.main, [*base, '--seed-range', '1', '2'])
            assert len(read_csv(out / 'runs.csv')) == 2
            assert next(out.glob('*seed1*_r1.json')).read_bytes() == original
            # Recover CSV from authoritative JSON after an interrupted CSV checkpoint.
            (out / 'runs.csv').unlink()
            invoke(run.main, [*base, '--seeds', '1', '2'])
            assert len(read_csv(out / 'runs.csv')) == 2
            invoke(run.main, [*base, '--seeds', '2', '--repeats', '2'])
            assert len(read_csv(out / 'runs.csv')) == 3
            saved = (out / 'runs.csv').read_bytes()
            rejected(run.main, [*base, '--seeds', '1', '--seconds', '2'])
            assert (out / 'runs.csv').read_bytes() == saved
        rejected(run.main, ['--datasets', 'long', '--seed-range', '1', '20', '--out', root / 'bad'])
        for obsolete in ['mnp', 'vertex_mn', 'path_rows25', 'path_probe25']:
            rejected(run.main, ['--methods', obsolete])
        # Dataset and algorithm failures are checkpointed without aborting the remaining runs.
        for failure in ('load', 'solve'):
            errors = root / failure
            calls = []
            def solve(*args, **kwargs):
                calls.append(args[1])
                raise RuntimeError('injected solver failure')
            with patch('run.load', side_effect=RuntimeError('injected dataset failure') if failure == 'load' else load), patch('run.solve', solve):
                rejected(run.main, ['--out', errors, '--datasets', 'speech', '--methods', 'path', 'kelley',
                                    '--solver', 'highs', '--seeds', '1'])
            records = read_csv(errors / 'runs.csv')
            assert len(records) == 2 and all(r['status'] == 'error' and r['error'] for r in records)
            assert all(r['upper'] == '' and r['queries'] == '' for r in records)
            if failure == 'solve':
                assert calls == ['path', 'kelley']
        # Synthetic paired observations exercise timeouts, missing values and failure accounting.
        synthetic = root / 'synthetic'; synthetic.mkdir()
        run.write_json(synthetic / 'config.json', {'arguments': {'atol': 1e-7, 'rtol': 1e-9}})
        rows = []
        for seed, method, status, duration, upper, lower in [
            (1, 'path', 'optimal', 1, -3, -3), (2, 'path', 'time_limit', 150, -2, -5),
            (1, 'kelley', 'optimal', 2, -3, -3), (2, 'kelley', 'error', 5, None, None),
        ]:
            rows.append(dict(dataset='toy', seed=seed, repeat=1, method=method, status=status,
                             seconds=duration, upper=upper, lower=lower, queries=10 if upper is not None else None,
                             lp_seconds=.1 if upper is not None else None, qp_seconds=None))
        run.write_csv(synthetic / 'runs.csv', rows)
        invoke(aggregate_results.main, ['--input', synthetic / 'runs.csv', '--out', synthetic / 'summary'])
        summaries = {r['method']: r for r in read_csv(synthetic / 'summary' / 'dataset_method.csv')}
        assert float(summaries['path']['mean_seconds']) == 75.5
        assert float(summaries['path']['mean_seconds_certified']) == 1
        assert float(summaries['path']['certification_percent']) == 50
        assert int(summaries['path']['timeouts']) == 1
        assert int(summaries['kelley']['failures']) == 1
        assert summaries['path']['mean_qp_seconds'] == ''
        assert summaries['path']['available_valid_lower'] == '2'
        assert len(read_csv(synthetic / 'summary' / 'paired_runs.csv')) == 4
        assert len(read_csv(synthetic / 'summary' / 'overall.csv')) == 2
    print('PASS resume/extension/repeats, config protection, error continuation, CLI and timeout/missing-value aggregation')


if __name__ == '__main__':
    main()
