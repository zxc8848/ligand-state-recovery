"""Holm correction over an explicit, fully present confirmatory file list."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--expected_tests', type=int, required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    rows = []
    for path in a.inputs:
        d = json.loads(Path(path).read_text(encoding='utf-8'))
        for ep, r in d['endpoints'].items():
            rows.append({'file': path, 'endpoint': ep, 'p_raw': r['mcnemar_p']})
    if len(rows) != a.expected_tests:
        raise ValueError(f'Expected family size {a.expected_tests}, found {len(rows)}')
    running = 0.
    for rank, row in enumerate(sorted(rows, key=lambda r: r['p_raw'])):
        running = max(running, min(1., row['p_raw']*(len(rows)-rank)))
        row['p_holm'] = running
    Path(a.output).write_text(json.dumps({'family_size': len(rows), 'tests': rows}, indent=1), encoding='utf-8')
    print('Holm family', len(rows))


if __name__ == '__main__': main()
