"""Report the two endpoints side by side, and stop conflating them.

There are two different "species recovery" numbers in this project and the
ledgers mixed them:

  FULL (B,h,q)   pose_localmap.py's own `*_species` field. ``joint.build_species``
                 constructs the molecule from the PREDICTED bond orders, the
                 PREDICTED formal charge (state//5-2) and the PREDICTED hydrogen
                 count (state%5) with NoImplicit set, then compares the canonical
                 SMILES against the reference heavy-only molecule WITH its own
                 charges and hydrogens. Self-consistency check: feeding the
                 reference B and reference (h,q) through the same builder
                 reproduces the reference SMILES for 488/488 development poses.

  CHARGE-BLIND   evaluate_species_recovery.py's `species_recovery`. It passes
                 formal_charge = [0]*n for BOTH sides and lets RDKit re-infer
                 hydrogens from valence (see _graph_candidate and _candidate_mol
                 docstrings). It therefore validates B only -- it cannot confirm
                 or refute a predicted charge or hydrogen count.

Both are legitimate; they answer different questions. The paper's task statement
("recover bond order, formal charge and implicit hydrogen from (Z,R,Q)") is only
addressed by the first.

IMPORTANT ASYMMETRY. The archived two-stage baseline cannot be scored on the full
endpoint at all: its per-edge dump carries no per-atom q/h, and the archived code
states the heavy-only refiner "intentionally has no direct charge/H supervision"
and "the solver leaves implicit_h at zero for every atom". So the full endpoint is
reportable as an ABSOLUTE number for this method, with no paired baseline. Every
head-to-head comparison in this project lives on the charge-blind endpoint.

Second limitation: the delocalized rewriting is applied to the per-edge dump,
after build_species has already run, so the full endpoint exists for greedy and
localmap but not for localmap_deloc.

Standard library only.
"""
import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ['ccd_sigma0p00', 'pdbbind_sigma0p00', 'pdbbind_sigma0p05',
         'ccd_sigma0p10', 'pdbbind_sigma0p10', 'pdbbind_sigma0p15']
SHORT = {'ccd_sigma0p00': 'CCD  ideal 0.00', 'pdbbind_sigma0p00': 'pose 0.00',
         'pdbbind_sigma0p05': 'pose 0.05', 'ccd_sigma0p10': 'CCD  0.10',
         'pdbbind_sigma0p10': 'pose 0.10', 'pdbbind_sigma0p15': 'pose 0.15'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(HERE/'joint_v6'))
    ap.add_argument('--seeds', default='seed20261002,seed20261003,seed20261004')
    ap.add_argument('--tag', default='c')
    ap.add_argument('--kind', default='fullgraph', choices=['fullgraph', 'refedges'])
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(',') if s.strip()]
    root = Path(a.root)
    out = {'seeds': seeds, 'conditions': {}}

    print('FULL (B,h,q) identity -- predicted bond orders, charges AND hydrogens.')
    print('No baseline exists on this endpoint: the archived two-stage does not solve for h.')
    print()
    print(f'{"condition":17s} {"n":>5s} {"greedy":>16s} {"localmap":>16s} {"gain pp":>12s}'
          f'   {"charge-blind localmap":>22s}')
    for c in ORDER:
        g, l, n = [], [], None
        blind = []
        for s in seeds:
            f = root/s/c/f'pose_localmap_{a.tag}.json'
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding='utf-8'))['summary']
            n = d['n']
            g.append(d['baseline_species'])
            l.append(d['localmap_species'])
            sp = root/s/c/('species_localmap_fullgraph.json' if a.kind == 'fullgraph' else 'species_localmap.json')
            if sp.exists():
                blind.append(json.loads(sp.read_text(encoding='utf-8'))['species_recovery'])
        if not g:
            print(f'{SHORT.get(c, c):17s}  -- not run --')
            continue
        sd = lambda v: st.stdev(v) if len(v) > 1 else 0.0
        gain = [100*(l[i]-g[i]) for i in range(len(g))]
        cell = (f'{st.mean(blind):.4f}±{sd(blind):.4f}' if blind else '--')
        print(f'{SHORT.get(c, c):17s} {n:5d} {st.mean(g):8.4f}±{sd(g):.4f} '
              f'{st.mean(l):8.4f}±{sd(l):.4f} {st.mean(gain):+7.2f}±{sd(gain):4.2f}'
              f'   {cell:>22s}')
        out['conditions'][c] = {
            'n': n, 'full_greedy': g, 'full_localmap': l,
            'full_gain_pp': gain, 'charge_blind_localmap': blind}
    print()
    print('The last column is the charge-blind endpoint for the same runs. The two')
    print('columns are NOT nested and must never be reported as one number: at pose')
    print('0.00 the full endpoint is higher, because zeroing charges can break a build')
    print('that the predicted charges would have completed, while H re-inference can')
    print('collapse two genuinely different species into one.')
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
