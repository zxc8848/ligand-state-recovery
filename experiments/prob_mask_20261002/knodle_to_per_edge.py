"""Convert Knodle's predictions into the same per-edge dump the other two use.

The problem this fixes. Until 2026-10-04 the three systems were not scored by the
same equivalence definition:

  evaluate_species_recovery.py  old two-stage and the new pipeline, via
                                classify() plus the charge-blind SMILES oracle
  score_knodle_vs_bondnet.py    Knodle, via
                                resonance_align_terminal_oxy(elements=(6,7,15,16))
                                with "no classify() fallback"

So putting Knodle's resonance figure in the same column as the other two, which
the ledger tables did, compares two different definitions. External review caught
this; it is my error, not the archive's -- each script is internally consistent.

The fix is not to edit either scorer. It is to put Knodle's output into the
per-edge format and run the SAME scorer on it, so one definition covers all
three systems.

Index space. preds_knodle.json is keyed by the molecule's position in
_read_valid_molecules(sdf) -- every molecule that parses and survives the MDL
aromaticity pass -- not by name. This module imports that function and
heavy_graph from the archive rather than reimplementing them, so the indexing
cannot drift from the adapt step that produced the predictions.

Labels: {'ar': 4, 1: 1, 2: 2, 3: 3} exactly as score_knodle_vs_bondnet._ORDER.

Molecules Knodle produced no output for get reference rows with pred = 0: the
predicted graph is then empty, the build fails and the molecule counts as not
recovered. That is the archived policy ("a molecule Knodle fails to convert is
scored as an incorrect prediction"), not a silent skip.

Emits the UNION of reference bonds and Knodle's predicted pairs, matching the
fullgraph convention, so off-reference false positives are visible.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'archive/legacy_two_stage_20261003'

ORDER = {'ar': 4, 1: 1, 2: 2, 3: 3, '1': 1, '2': 2, '3': 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--knodle_pred', required=True)
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--sigma', type=float, default=0.0)
    ap.add_argument('--report', default='')
    a = ap.parse_args()

    sys.path.insert(0, str(LEGACY))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from scripts.p0c_yuelbond_compare import _read_valid_molecules
    from scripts.benchmark_templatefree_motifs import heavy_graph

    preds = json.loads(Path(a.knodle_pred).read_text(encoding='utf-8'))
    meta = preds.get('__meta__', {})
    mols = _read_valid_molecules(a.reference_sdf)
    if meta.get('n_molecules') not in (None, len(mols)):
        raise SystemExit(f'index space mismatch: predictions were made for '
                         f'{meta["n_molecules"]} molecules, this SDF yields {len(mols)}. '
                         f'Re-run the adapt step against the SDF you intend to score.')

    lines, missing, off_ref, n = [], [], 0, 0
    unknown = set()
    for idx, mol in enumerate(mols):
        heavy = Chem.RemoveHs(Chem.Mol(mol), sanitize=True)
        if heavy.GetNumConformers() == 0:
            continue
        name = heavy.GetProp('_Name') if heavy.HasProp('_Name') else f'mol{idx}'
        truth = heavy_graph(heavy)
        z = {at.GetIdx(): at.GetAtomicNum() for at in heavy.GetAtoms()}
        raw = preds.get(str(idx))
        if raw is None:
            missing.append(name)
            pred = {}
        else:
            pred = {}
            for i, j, lab in raw:
                if lab not in ORDER:
                    unknown.add(str(lab))
                    continue
                pred[tuple(sorted((int(i), int(j))))] = ORDER[lab]
        n += 1
        for e in sorted(set(truth) | set(pred)):
            t, p = truth.get(e, 0), pred.get(e, 0)
            row = {'name': name, 'sigma': a.sigma, 'i': e[0], 'j': e[1],
                   'z_i': z[e[0]], 'z_j': z[e[1]], 'true': t, 'pred': p,
                   'correct': int(p == t)}
            if e not in truth:
                row['off_reference'] = True
                off_ref += 1
            lines.append(json.dumps(row))
    Path(a.output).write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'molecules {n}   rows {len(lines)}   off-reference Knodle bonds {off_ref}')
    print(f'no Knodle output (scored as not recovered): {len(missing)}'
          + (f'   {missing[:6]}' if missing else ''))
    if unknown:
        raise SystemExit(f'unmapped Knodle bond labels {sorted(unknown)}; refusing to '
                         f'guess -- extend ORDER deliberately')
    if a.report:
        Path(a.report).write_text(json.dumps(
            {'knodle_pred': a.knodle_pred, 'reference_sdf': a.reference_sdf,
             'molecules': n, 'rows': len(lines), 'off_reference_bonds': off_ref,
             'no_output': missing,
             'note': 'Knodle predictions in the shared per-edge format so that one '
                     'scorer covers all three systems; union of reference bonds and '
                     'predicted pairs'}, indent=1), encoding='utf-8')
        print('wrote', a.report)


if __name__ == '__main__':
    main()
