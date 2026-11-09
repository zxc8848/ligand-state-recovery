"""Can the reference graph recover itself under the graph-SMILES endpoint?

The graph canonical-SMILES endpoint zeroes formal charges on BOTH sides and lets
RDKit re-infer hydrogens, then requires the two strings to match, counting an
unbuildable graph on either side as a miss. Section 'What counts as recovery'
reports a 488/488 self-consistency check for the NATIVE endpoint only. The
graph-SMILES endpoint has no such check, and it cannot be assumed: the reference
bond graph, stripped of its charges, need not sanitize.

If it does not, the endpoint has a ceiling below 1.0 that is a property of the
metric rather than of any model, and a reader cannot separate model error from
metric construction. This script measures that ceiling and classifies the
molecules that fail it. It runs no model and changes nothing.

    python reference_selfscore.py --reference_sdf <sdf> [--population_manifest <json>]
                                  [--output <json>]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'archive/legacy_two_stage_20261003'
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--population_manifest', default='')
    ap.add_argument('--output', default='')
    a = ap.parse_args()

    sys.path.insert(0, str(LEGACY))
    from evaluation_common import reference_population, ALLOWED
    from scripts.evaluate_species_recovery import _graph_candidate
    from scripts.evaluate_candidate_sets import species_signature
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    refs, excluded = reference_population(a.reference_sdf, ALLOWED)
    if a.population_manifest:
        from population_manifest import validate_population
        checked, _ = validate_population(a.reference_sdf, a.population_manifest)
        if set(refs) != set(checked):
            raise SystemExit('population differs from the frozen manifest')

    ok, failed = 0, []
    for name, mol in refs.items():
        truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                 4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())
                 for b in mol.GetBonds()}
        adjacency = collections.defaultdict(set)
        for i, j in truth:
            adjacency[i].add(j); adjacency[j].add(i)
        elems = [at.GetAtomicNum() for at in mol.GetAtoms()]
        rk, _, status = species_signature(mol, _graph_candidate(elems, truth),
                                          adjacency, return_status=True)
        if rk.startswith('fallback::'):
            # why: record the chemistry so the ceiling is interpretable
            charged = [at.GetSymbol() for at in mol.GetAtoms() if at.GetFormalCharge()]
            failed.append({
                'name': name, 'status': str(status),
                'n_heavy': mol.GetNumAtoms(),
                'net_charge': sum(at.GetFormalCharge() for at in mol.GetAtoms()),
                'charged_atoms': collections.Counter(charged),
                'has_aromatic': any(b.GetIsAromatic() for b in mol.GetBonds()),
                'elements': sorted({at.GetSymbol() for at in mol.GetAtoms()}),
            })
        else:
            ok += 1

    n = len(refs)
    print(f'reference self-score on the graph-SMILES endpoint: {ok}/{n} = {ok/n:.4f}')
    print(f'ceiling imposed by the metric: {ok/n:.4f}  (a perfect bond-graph '
          f'prediction cannot exceed this)')
    if failed:
        print(f'\n{len(failed)} reference molecules do not round-trip:')
        by_charge = collections.Counter(f['net_charge'] for f in failed)
        print(f'  by reference net charge: {dict(sorted(by_charge.items()))}')
        el = collections.Counter(e for f in failed for e in f['elements'])
        print(f'  elements present: {dict(el.most_common(12))}')
        arom = sum(f['has_aromatic'] for f in failed)
        print(f'  containing aromatic bonds: {arom}/{len(failed)}')
        print('  names:', ', '.join(sorted(f['name'] for f in failed)[:40]))
    out = {'reference_sdf': str(a.reference_sdf), 'n': n, 'self_score': ok/n,
           'passed': ok, 'failed': failed}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
