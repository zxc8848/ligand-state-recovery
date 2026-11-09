"""Aggregate unified_resonance.py over systems x conditions x training seeds.

Prints one table per endpoint with all three systems side by side under ONE
equivalence definition, and the shift each system takes relative to the
definition it was previously scored with.

Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path
from evaluation_common import load_verdicts

HERE = Path(__file__).resolve().parent
ORDER = ['ccd_sigma0p00', 'pdbbind_sigma0p00', 'pdbbind_sigma0p05',
         'ccd_sigma0p10', 'pdbbind_sigma0p10', 'pdbbind_sigma0p15']
SHORT = {'ccd_sigma0p00': 'CCD ideal 0.00', 'pdbbind_sigma0p00': 'pose 0.00',
         'pdbbind_sigma0p05': 'pose 0.05', 'ccd_sigma0p10': 'CCD  0.10',
         'pdbbind_sigma0p10': 'pose 0.10', 'pdbbind_sigma0p15': 'pose 0.15'}


def load(p):
    return load_verdicts(p)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(HERE/'joint_v6'))
    ap.add_argument('--seeds', default='seed20261002,seed20261003,seed20261004')
    ap.add_argument('--conditions', default=','.join(ORDER))
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(',') if s.strip()]
    conds = [c.strip() for c in a.conditions.split(',') if c.strip()]
    root = Path(a.root)
    out = {'seeds': seeds, 'conditions': {}}

    print('UNIFIED resonance-aware recovery: one definition for all three systems')
    print('(terminal-oxy alignment then chemically validated aromatic components)')
    print()
    print(f'{"condition":15s} {"n":>5s} {"Knodle":>18s} {"old two-stage":>18s}'
          f' {"new +deloc":>18s} {"new - old":>11s} {"new - Knodle":>13s}')
    for c in conds:
        kn = load(root/'unified'/c/'knodle.json')
        old = load(root/'unified'/c/'old.json')
        new = [load(root/'unified'/c/f'new_{s}.json') for s in seeds]
        populations = [{r['name'] for r in x['per_molecule']} for x in [kn, old]+new]
        if any(p != populations[0] for p in populations[1:]):
            raise ValueError(f'{c}: method/seed populations differ')
        nv = [x['unified_resonance'] for x in new]
        sd = st.stdev(nv) if len(nv) > 1 else 0.0
        print(f'{SHORT.get(c, c):15s} {kn["n"]:5d} {kn["unified_resonance"]:18.4f}'
              f' {old["unified_resonance"]:18.4f} {st.mean(nv):11.4f}±{sd:.4f}'
              f' {100*(st.mean(nv)-old["unified_resonance"]):+11.2f}'
              f' {100*(st.mean(nv)-kn["unified_resonance"]):+13.2f}')
        out['conditions'][c] = {
            'n': kn['n'], 'knodle': kn['unified_resonance'],
            'old': old['unified_resonance'], 'new_per_seed': nv,
            'knodle_classify': kn['classify_resonance'],
            'old_classify': old['classify_resonance'],
            'new_classify': [x['classify_resonance'] for x in new]}
    print()
    print('Diagnostic difference from classify() on THE SAME predictions, in pp:')
    print(f'{"condition":15s} {"Knodle":>10s} {"old":>10s} {"new":>10s}')
    for c in conds:
        r = out['conditions'].get(c)
        if not r:
            continue
        print(f'{SHORT.get(c, c):15s} '
              f'{100*(r["knodle"]-r["knodle_classify"]):10.2f} '
              f'{100*(r["old"]-r["old_classify"]):10.2f} '
              f'{100*(st.mean(r["new_per_seed"])-st.mean(r["new_classify"])):10.2f}')
    print()
    print('This diagnostic compares unified flags with legacy classify, not with')
    print('the Knodle rates previously published using terminal-oxy alignment.')
    print('A large Knodle difference and a near-zero difference for the other two is the')
    print('expected signature: classify() accepts class-5 but not all-aromatic, so it')
    print('forgave the old pipeline\'s convention and penalised Knodle\'s. Scoring')
    print('everything with classify() would have understated the Knodle baseline by')
    print('roughly 30pp -- the opposite of a fair unification.')
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
