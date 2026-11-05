"""CSV summaries; missing metrics stay missing and timeouts remain observations."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics

from run import write_csv


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def statistic(rows, field, operation=statistics.mean):
    values = [v for r in rows if (v := number(r.get(field))) is not None]
    return operation(values) if values else None


def deviation(values):
    return statistics.stdev(values) if len(values) > 1 else None


def summarize(rows):
    successful = [r for r in rows if r['certified']]
    result = dict(
        runs=len(rows), certified=len(successful), certification_percent=100 * len(successful) / len(rows),
        timeouts=sum(r['status'] == 'time_limit' for r in rows),
        failures=sum(r['status'] not in ('optimal', 'time_limit', 'iteration_limit') or
                     (r['status'] == 'optimal' and not r['certified']) for r in rows),
        iteration_limits=sum(r['status'] == 'iteration_limit' for r in rows),
        mean_seconds=statistic(rows, 'seconds'), median_seconds=statistic(rows, 'seconds', statistics.median),
        std_seconds=statistic(rows, 'seconds', deviation),
        mean_seconds_certified=statistic(successful, 'seconds'),
        mean_queries=statistic(rows, 'queries'), std_queries=statistic(rows, 'queries', deviation),
    )
    for field in ('oracle_seconds', 'lp_seconds', 'qp_seconds', 'iterations', 'upper',
                  'valid_lower', 'certified_gap', 'gap_to_best_incumbent', 'runtime_rank'):
        result['mean_' + field] = statistic(rows, field)
    for field in ('seconds', 'queries', 'valid_lower', 'certified_gap', 'gap_to_best_incumbent'):
        result['available_' + field] = sum(number(r.get(field)) is not None for r in rows)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='runs.csv')
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    config = json.loads((args.input.parent / 'config.json').read_text(encoding='utf-8'))
    settings = config['arguments']
    with args.input.open(newline='', encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        p.error('No results')
    instances = defaultdict(list)
    groups = defaultdict(list)
    for r in rows:
        upper, lower = number(r.get('upper')), number(r.get('lower'))
        tolerance = (settings['atol'] + settings['rtol'] * max(1, abs(upper), abs(lower))
                     if upper is not None and lower is not None else None)
        valid = tolerance is not None and lower <= upper + 10 * tolerance
        r['valid_lower'] = lower if valid else None
        r['certified_gap'] = max(0, upper - lower) if valid else None
        r['certified'] = bool(r['status'] == 'optimal' and valid and upper - lower <= tolerance)
        instances[r['dataset'], r['seed'], r.get('fingerprint', '')].append(r)
        groups[r['dataset'], r['method']].append(r)
    for selected in instances.values():
        best = min((v for r in selected if (v := number(r.get('upper'))) is not None), default=None)
        for r in selected:
            upper = number(r.get('upper'))
            r['gap_to_best_incumbent'] = upper - best if upper is not None and best is not None else None
        # Rank within each repetition: certification first, then observed elapsed time.
        repeats = defaultdict(list)
        for r in selected:
            repeats[r['repeat']].append(r)
        for paired in repeats.values():
            def score(r):
                duration = number(r.get('seconds'))
                return (not r['certified'], duration if duration is not None else math.inf)
            scores = sorted(score(r) for r in paired)
            for r in paired:
                matches = [i + 1 for i, s in enumerate(scores) if s == score(r)]
                r['runtime_rank'] = statistics.mean(matches)
    args.out.mkdir(parents=True, exist_ok=True)
    summaries = [dict(dataset=d, method=m, **summarize(selected))
                 for (d, m), selected in sorted(groups.items())]
    write_csv(args.out / 'dataset_method.csv', summaries)
    methods = sorted({r['method'] for r in rows})
    overall = []
    for method in methods:
        selected = [r for r in rows if r['method'] == method]
        overall.append(dict(method=method, runs=len(selected),
                            certification_percent=100 * sum(r['certified'] for r in selected) / len(selected),
                            mean_paired_rank=statistic(selected, 'runtime_rank')))
    overall.sort(key=lambda r: (-r['certification_percent'], r['mean_paired_rank']))
    write_csv(args.out / 'overall.csv', [dict(rank=i, **r) for i, r in enumerate(overall, 1)])
    write_csv(args.out / 'paired_runs.csv', rows)
    print(f'CSV summaries saved to {args.out.resolve()}')


if __name__ == '__main__':
    main()
