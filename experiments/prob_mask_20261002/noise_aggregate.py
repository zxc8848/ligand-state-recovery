"""Aggregate the noise / CCD / Knodle sweep into the tables the paper needs.

Reads joint_v6/<ckpt>/<condition>/{species_*,comparison_vs_old_*,knodle_vs_new_*}.json
and prints three tables: absolute accuracies, paired deltas vs the archived
two-stage pipeline, and paired deltas vs Knodle. Standard library only.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDS = ['pdbbind_sigma0p00', 'pdbbind_sigma0p05', 'pdbbind_sigma0p10',
         'pdbbind_sigma0p15', 'ccd_sigma0p00', 'ccd_sigma0p10']
SELS = ['greedy', 'localmap', 'localmap_deloc']
SHORT = {'pdbbind_sigma0p00': 'pose 0.00', 'pdbbind_sigma0p05': 'pose 0.05',
         'pdbbind_sigma0p10': 'pose 0.10', 'pdbbind_sigma0p15': 'pose 0.15',
         'ccd_sigma0p00': 'CCD  0.00', 'ccd_sigma0p10': 'CCD  0.10'}
KEYS = [('strict', 'strict'), ('resonance_aware', 'resonance'),
        ('canonical_smiles', 'SMILES')]


def jload(p):
    return json.loads(Path(p).read_text(encoding='utf-8')) if Path(p).exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='joint_v1')
    ap.add_argument('--scope', default='all', choices=['all', 'geom_disjoint'])
    ap.add_argument('--kind', default='fullgraph', choices=['fullgraph', 'refedges'],
                    help='which dump the scorer outputs came from; fullgraph is '
                         'primary, refedges is the published-protocol variant')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    # the fullgraph outputs carry a suffix; the published-protocol ones do not
    suf = '' if a.kind == 'refedges' else '_fullgraph'
    root = HERE/'joint_v6'/a.ckpt
    out = {'ckpt': a.ckpt, 'scope': a.scope, 'conditions': {}}

    print(f'checkpoint {a.ckpt}   scope {a.scope}')
    print()
    print('TABLE 1  absolute accuracy (strict / resonance-aware / canonical-SMILES)')
    print(f'{"condition":11s} {"n":>5s}  {"system":22s} {"strict":>8s} {"reson":>8s} {"SMILES":>8s}')
    for c in CONDS:
        d = root/c
        cmp_d = jload(d/'comparison_vs_old_localmap_deloc.json')
        kn = jload(d/'knodle_vs_new_localmap_deloc.json')
        if cmp_d is None:
            print(f'{SHORT[c]:11s}  -- not run --'); continue
        sc = cmp_d[a.scope]
        n = sc['n']
        rec = {'n': n}
        rows = [('archived two-stage + MILP',
                 sc['strict']['old'], sc['resonance_aware']['old'],
                 sc['canonical_smiles']['old'])]
        if kn and a.scope == 'all':
            rows.append(('Knodle (template rules)',
                         kn['knodle']['strict'], kn['knodle']['resonance'],
                         kn['knodle']['species']))
        for s in SELS:
            sp = jload(d/f'species_{s}{suf}.json')
            if sp is None:
                continue
            strict = jload(d/f'comparison_vs_old_{s}{suf}.json')[a.scope]['strict']['new']
            rows.append((f'new / {s}', strict,
                         sp['classify_recovery'], sp['species_recovery']))
        for i, (name, st, re_, sm) in enumerate(rows):
            head = SHORT[c] if i == 0 else ''
            nn = f'{n:5d}' if i == 0 else '     '
            print(f'{head:11s} {nn}  {name:22s} {st:8.4f} {re_:8.4f} {sm:8.4f}')
            rec[name] = {'strict': st, 'resonance': re_, 'SMILES': sm}
        print()
        out['conditions'][c] = rec

    for label, fname, getter in [
        ('TABLE 2  paired delta, new(+deloc) minus archived two-stage, same molecules',
         'comparison_vs_old_localmap_deloc.json', 'old'),
        ('TABLE 3  paired delta, new(+deloc) minus Knodle, same molecules',
         'knodle_vs_new_localmap_deloc.json', 'knodle')]:
        print(label)
        print(f'{"condition":11s} {"endpoint":10s} {"delta":>8s} {"CI95":>20s} {"McNemar p":>12s}'
              f' {"disc new/other":>16s}')
        for c in CONDS:
            d = jload(root/c/fname)
            if d is None:
                continue
            if getter == 'old':
                sc = d[a.scope]
                for k, lab in KEYS:
                    e = sc[k]
                    ci = e.get('ci95') or [float('nan')]*2
                    print(f'{SHORT[c] if lab=="strict" else "":11s} {lab:10s} '
                          f'{100*e["delta"]:+8.2f} '
                          f'{f"[{100*ci[0]:+.2f},{100*ci[1]:+.2f}]":>20s} '
                          f'{e["mcnemar_p"]:12.2e} '
                          f'{str(e["discordant_new_only"])+"/"+str(e["discordant_old_only"]):>16s}')
            else:
                if a.scope != 'all':
                    continue
                for k, lab in [('strict', 'strict'), ('resonance', 'resonance'),
                               ('species', 'SMILES')]:
                    delta = d['bondnet'][k] - d['knodle'][k]
                    pt = d['paired_tests']['strict' if k == 'strict' else
                                           ('resonance' if k == 'resonance' else 'species')]
                    print(f'{SHORT[c] if lab=="strict" else "":11s} {lab:10s} '
                          f'{100*delta:+8.2f} {"":>20s} '
                          f'{pt["mcnemar_exact_p"]:12.2e} '
                          f'{str(pt["bondnet_only"])+"/"+str(pt["knodle_only"]):>16s}')
            print()
        print()

    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
