"""Split every comparison by whether the REFERENCE has a delocalizable group.

Why this exists. The reference files contain zero class-5 (delocalized) edges,
but the archived two-stage pipeline emits class-5 on 114/488 poses and 227/1000
CCD molecules. So the `strict` endpoint penalises that labelling convention
(every class-5 edge is an exact-label error) while the `resonance-aware` endpoint
forgives it wholesale -- on the 212 CCD molecules where only the old pipeline
used class-5, its strict accuracy is 0.0000 and its resonance accuracy 0.9623.
Neither endpoint is measuring bond perception there; both are measuring agreement
with a convention.

Splitting on the PREDICTIONS would be a post-hoc, prediction-conditioned subset.
So the split here is defined from the reference graph only: a molecule carries a
delocalizable group iff some non-oxygen centre bears >= 2 oxygens that each have
exactly one reference bond (carboxylate, phosphate, nitro, sulfonyl, ...).

Consistency check printed at the end: on the no-delocalizable-group subset the
strict and resonance-aware deltas must be identical, because the resonance scorer
then has nothing to forgive. If they differ, this split is wrong.

Standard library only.
"""
import argparse
import collections
import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def reference_delocalizable(per_edge):
    """{names with a delocalizable group}, from reference labels only."""
    adj = collections.defaultdict(lambda: collections.defaultdict(list))
    z = {}
    for line in Path(per_edge).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['true'] == 0:
            continue
        adj[r['name']][r['i']].append(r['j'])
        adj[r['name']][r['j']].append(r['i'])
        z[(r['name'], r['i'])] = r['z_i']
        z[(r['name'], r['j'])] = r['z_j']
    flagged = set()
    for name, a in adj.items():
        for centre, nb in a.items():
            if z[(name, centre)] == 8:
                continue
            if sum(1 for x in nb if z[(name, x)] == 8 and len(a[x]) == 1) >= 2:
                flagged.add(name)
                break
    return flagged


def strict_by_molecule(per_edge):
    ok = {}
    for line in Path(per_edge).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        ok[r['name']] = ok.get(r['name'], True) and bool(r['correct'])
    return ok


def verdicts(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return {r['name']: r for r in d['per_molecule']}


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if y and not x)
    c = sum(1 for x, y in pairs if x and not y)
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    return min(1.0, sum(math.comb(n, i) for i in range(k+1))/2**n*2), b, c


def bootstrap(pairs, reps=2000, seed=11):
    rng = random.Random(seed)
    m = len(pairs)
    ds = []
    for _ in range(reps):
        s = [pairs[rng.randrange(m)] for _ in range(m)]
        ds.append(sum(y for _, y in s)/m - sum(x for x, _ in s)/m)
    ds.sort()
    return ds[int(.025*reps)], ds[int(.975*reps)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default='pdbbind_sigma0p00,pdbbind_sigma0p05,ccd_sigma0p00')
    ap.add_argument('--ckpt', default='joint_v1')
    ap.add_argument('--selector', default='localmap_deloc')
    ap.add_argument('--kind', default='fullgraph', choices=['fullgraph', 'refedges'],
                    help='which dump to score. fullgraph = reference edges UNION '
                         'every non-zero prediction (the honest full-graph '
                         'endpoint, and symmetric with the archived dump, which '
                         'edge_records() already wrote as "one row per edge that '
                         'is either a true bond or a prediction"). refedges = '
                         'projection onto the reference edge list, kept only for '
                         'comparability with the published numbers.')
    ap.add_argument('--probe', default=str(ROOT/'work/paper2_noise_ccd_probe'))
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    out = {'ckpt': a.ckpt, 'selector': a.selector, 'kind': a.kind,
           'conditions': {}}

    for cond in [c.strip() for c in a.conditions.split(',') if c.strip()]:
        E = HERE/'joint_v6'/a.ckpt/cond
        old_edge = Path(a.probe)/cond/'bondnet_per_edge.jsonl'
        new_edge = E/f'pose_{a.selector}_c_{a.kind}.jsonl'
        for need in (old_edge, new_edge, E/'species_old.json',
                     E/f'species_{a.selector}{"" if a.kind == "refedges" else "_fullgraph"}.json'):
            if not Path(need).exists():
                raise SystemExit(f'missing {need}')
        flagged = reference_delocalizable(old_edge)
        so, sn = strict_by_molecule(old_edge), strict_by_molecule(new_edge)
        vo = verdicts(E/'species_old.json')
        vn = verdicts(E/f'species_{a.selector}'
                      f'{"" if a.kind == "refedges" else "_fullgraph"}.json')
        names = sorted(set(vo) & set(vn) & set(so) & set(sn))
        print(f'===== {cond}   n={len(names)}   '
              f'reference has a delocalizable group: {len(flagged & set(names))}')
        rows = {}
        for lab, sub in (('all (published)', names),
                         ('reference: no deloc group', [n for n in names if n not in flagged]),
                         ('reference: has deloc group', [n for n in names if n in flagged])):
            if len(sub) < 20:
                continue
            cells, rec = [], {'n': len(sub)}
            for tag, key in (('strict', None), ('reson', 'classify_species'),
                             ('SMILES', 'smiles_species')):
                pairs = ([(int(so[n]), int(sn[n])) for n in sub] if key is None
                         else [(int(vo[n][key]), int(vn[n][key])) for n in sub])
                d = (sum(y for _, y in pairs) - sum(x for x, _ in pairs))/len(sub)
                p, b, c = mcnemar(pairs)
                lo, hi = bootstrap(pairs)
                cells.append(f'{tag} {100*d:+6.2f} [{100*lo:+.1f},{100*hi:+.1f}] p={p:.1e}')
                rec[tag] = {'delta': d, 'ci95': [lo, hi], 'p': p,
                            'new_only': b, 'old_only': c,
                            'old': sum(x for x, _ in pairs)/len(sub),
                            'new': sum(y for _, y in pairs)/len(sub)}
            rows[lab] = rec
            print(f'  {lab:28s} n={len(sub):4d}  ' + ' | '.join(cells))
        nod = rows.get('reference: no deloc group')
        if nod:
            same = abs(nod['strict']['delta'] - nod['reson']['delta']) < 1e-12
            print(f'  consistency: strict == resonance on the no-deloc subset: {same}'
                  + ('' if same else '   <- SPLIT IS WRONG, investigate'))
        out['conditions'][cond] = rows
        print()
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
