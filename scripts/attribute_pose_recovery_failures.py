"""Descriptive failure taxonomy and fixed-output reference substitutions."""

import argparse
from collections import Counter
import json
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

from evaluate_pose_external_states import build, reference_population


def graph_of(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
            4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}


def taxonomy(ref, prediction, target, truth, graph, h, q):
    native = prediction['good'] and prediction['smiles'] == target
    ref_h = [a.GetTotalNumHs() for a in ref.GetAtoms()]
    ref_q = [a.GetFormalCharge() for a in ref.GetAtoms()]
    if native:
        return 'native_match'
    if not prediction['good']:
        return 'unconstructible'
    if set(graph) != set(truth):
        return 'connectivity'
    if graph != truth:
        return 'aromatic_bond_labels' if any(4 in (graph[e], truth[e]) for e in truth if graph[e] != truth[e]) else 'other_bond_order'
    if h != ref_h and q != ref_q:
        return 'hydrogens_and_charges'
    if q != ref_q:
        return 'charges_only'
    if h != ref_h:
        return 'hydrogens_only'
    raise ValueError(f"Unexplained native failure for {prediction['id']}")


def tautomer_match(a, b, enumerator):
    if a is None or b is None or Chem.GetFormalCharge(a) != Chem.GetFormalCharge(b):
        return False
    if rdMolDescriptors.CalcMolFormula(a) != rdMolDescriptors.CalcMolFormula(b):
        return False
    return Chem.MolToSmiles(enumerator.Canonicalize(a), isomericSmiles=False) == Chem.MolToSmiles(enumerator.Canonicalize(b), isomericSmiles=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--svg', type=Path)
    p.add_argument('--png', type=Path)
    p.add_argument('--n', type=int, default=491)
    args = p.parse_args()
    RDLogger.DisableLog('rdApp.*')
    refs, excluded = reference_population(args.sdf)
    if excluded or len(refs) != args.n:
        raise ValueError('Population mismatch')
    enumerator = rdMolStandardize.TautomerEnumerator()
    results = []
    for path in args.predictions:
        records = json.loads(path.read_text(encoding='utf-8'))['records']
        if len(records) != len(refs) or {r['id'] for r in records} != set(refs):
            raise ValueError('Predictions do not match declared population')
        rows = []
        for pred in records:
            name = pred['id']
            ref = refs[name]
            target = Chem.MolToSmiles(ref, isomericSmiles=False)
            truth, graph = graph_of(ref), {tuple(e): y for e, y in zip(pred['pairs'], pred['bond_labels'], strict=True) if y}
            z = [a.GetAtomicNum() for a in ref.GetAtoms()]
            ref_h, ref_q = [a.GetTotalNumHs() for a in ref.GetAtoms()], [a.GetFormalCharge() for a in ref.GetAtoms()]
            h, q = pred['h'], pred['q']
            s, good, reason = build(z, graph, h, q)
            if good != pred['good'] or (good and s != pred['smiles']):
                raise ValueError(f'{name}: independent build differs from saved output')
            category = taxonomy(ref, pred, target, truth, graph, h, q)
            a, ag, _ = build(z, truth, h, q)
            b, bg, _ = build(z, graph, ref_h, ref_q)
            if category != 'native_match' and good and set(graph) == set(truth):
                related = tautomer_match(ref, Chem.MolFromSmiles(s), enumerator)
            else:
                related = False
            rows.append(dict(name=name, primary=category, build_error=reason, target=target, predicted=s,
                             q_vector_correct=q == ref_q, h_vector_correct=h == ref_h, tautomer_related=related,
                             reference_B_with_selected_atoms=bool(ag and a == target),
                             selected_B_with_reference_atoms=bool(bg and b == target)))
        counter = dict(Counter(r['primary'] for r in rows))
        failed = [r for r in rows if r['primary'] != 'native_match']
        stats = dict(head=path.stem, n=len(rows), counts=counter, tautomer_related_failures=sum(r['tautomer_related'] for r in failed),
                     failures_rescued_by_reference_B=sum(r['reference_B_with_selected_atoms'] for r in failed),
                     failures_rescued_by_reference_states=sum(r['selected_B_with_reference_atoms'] for r in failed))
        print(stats)
        results.append(dict(**stats, records=rows))
    args.output.write_text(json.dumps(dict(note='Post hoc hierarchical categories and output substitutions; not oracle upper bounds.', heads=results), indent=2)+'\n', encoding='utf-8')
    if args.svg or args.png:
        # Use the first declared head, then the smallest reference among each
        # category for a readable figure. No hand-picked best/worst result.
        rows = results[0]['records']
        groups = [('other_bond_order', 'Bond order'), ('aromatic_bond_labels', 'Aromatic bond label'), ('hydrogens_and_charges', 'Hydrogen and charge')]
        reference_mols, predicted_mols, reference_titles, predicted_titles = [], [], [], []
        for key, label in groups:
            examples = [r for r in rows if r['primary'] == key and r['predicted'] is not None]
            if not examples:
                continue
            row = min(examples, key=lambda x: (refs[x['name']].GetNumAtoms(), x['name']))
            reference_mols.append(Chem.MolFromSmiles(row['target']))
            predicted_mols.append(Chem.MolFromSmiles(row['predicted']))
            reference_titles.append(f"{row['name']} | {label} | CCD")
            predicted_titles.append(f"{row['name']} | Prediction")
        common = dict(molsPerRow=len(reference_mols), subImgSize=(420, 235),
                      legends=reference_titles+predicted_titles, legendFontSize=20)
        if args.svg:
            svg = Draw.MolsToGridImage(reference_mols+predicted_mols, useSVG=True, **common)
            args.svg.write_text(svg, encoding='utf-8')
        if args.png:
            img = Draw.MolsToGridImage(reference_mols+predicted_mols, **common)
            img.save(args.png, dpi=(300, 300))


if __name__ == '__main__':
    main()
