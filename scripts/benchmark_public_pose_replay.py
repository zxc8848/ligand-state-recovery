"""Export Z/R/Q, score OpenBabel under the shared graph protocol, and score Stage C."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
from evaluation_common import reference_population, score_dump


def openbabel_graph(z, xyz):
    from openbabel import openbabel as ob
    molecule = ob.OBMol()
    for elem, coord in zip(z, xyz, strict=True):
        atom = molecule.NewAtom()
        atom.SetAtomicNum(elem)
        atom.SetVector(*[float(v) for v in coord])
    molecule.ConnectTheDots()
    molecule.PerceiveBondOrders()
    return {tuple(sorted((b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1))):
            4 if b.IsAromatic() else int(b.GetBondOrder()) for b in ob.OBMolBondIter(molecule)}


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--outdir', type=Path, required=True)
    p.add_argument('--expected-n', type=int, default=491)
    p.add_argument('--predictions', type=Path, nargs='+', help='If provided, score existing Stage C predictions instead of preparing inputs and running OpenBabel.')
    p.add_argument('--historical', type=Path, nargs='+', help='Optional matching historical head dumps for molecule-level parity.')
    args = p.parse_args()
    RDLogger.DisableLog('rdApp.*')
    refs, excluded = reference_population(args.sdf)
    if excluded or len(refs) != args.expected_n:
        raise ValueError('Unexpected pose population')
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.predictions:
        results = []
        target = {name: Chem.MolToSmiles(mol, isomericSmiles=False) for name, mol in refs.items()}
        if args.historical and len(args.historical) != len(args.predictions):
            raise ValueError('Provide one historical file per public prediction file')
        for index, path in enumerate(args.predictions):
            rows = json.loads(path.read_text(encoding='utf-8'))['records']
            if len(rows) != len(refs) or {r['id'] for r in rows} != set(refs):
                raise ValueError(f'Wrong prediction population in {path}')
            matches = {r['id']: bool(r['good'] and r['smiles'] == target[r['id']]) for r in rows}
            result = dict(run=path.name, n=len(rows), constructible=sum(bool(r['good']) for r in rows),
                          identity_matches=sum(matches.values()), per_molecule=matches)
            if args.historical:
                old = {r['name']: r for r in json.loads(args.historical[index].read_text(encoding='utf-8'))['records']}
                if set(old) != set(refs):
                    raise ValueError('Historical prediction population differs')
                old_selected = {k: r['candidates_detail'][r['selected_indices']['localmap']] for k, r in old.items()}
                result.update(changed_native_ids=[k for k, ok in matches.items() if ok != bool(old[k]['localmap_species'])],
                              changed_buildable_ids=[r['id'] for r in rows if bool(r['good']) != bool(old[r['id']]['localmap_valid'])],
                              changed_selected_state_ids=[r['id'] for r in rows if r['pairs'] != old[r['id']]['pairs'] or
                                                          any(r[field] != old_selected[r['id']][field] for field in ('bond_labels', 'atom_states'))])
            results.append(result)
        write_json(args.outdir/'native_summary.json', results)
        print([(r['run'], r['identity_matches'], r['constructible'], r['n']) for r in results])
        return
    from openbabel import openbabel as ob
    model_inputs, failures = [], {}
    edges = args.outdir/'openbabel_edges.jsonl'
    with edges.open('w', encoding='utf-8') as output:
        for name, mol in refs.items():
            z = [a.GetAtomicNum() for a in mol.GetAtoms()]
            xyz = mol.GetConformer().GetPositions().astype(np.float32)
            model_inputs.append(dict(id=name, Z=z, R=xyz.tolist(), Q=Chem.GetFormalCharge(mol)))
            try:
                pred = openbabel_graph(z, xyz)  # Native geometry-only route; no Q, bonds or H supplied.
                if not set(pred.values()) <= {1, 2, 3, 4}:
                    raise ValueError('Unsupported OpenBabel bond order')
            except Exception as exc:
                failures[name] = f'{type(exc).__name__}: {exc}'
                continue
            truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                     4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}
            for i, j in sorted(pred.keys() | truth.keys()):
                output.write(json.dumps(dict(name=name, i=i, j=j, true=truth.get((i, j), 0), pred=pred.get((i, j), 0)))+'\n')
    write_json(args.outdir/'inputs_zrq.json', model_inputs)
    scored = score_dump(edges, args.sdf, expected_n=args.expected_n)
    rows = scored['per_molecule']
    scored.update(method='OpenBabel ConnectTheDots + PerceiveBondOrders (Z,R only)', version=ob.OBReleaseVersion(),
                  failures=failures, typed_recall=sum(r['true_edge_hits'] for r in rows)/sum(r['true_edge_count'] for r in rows))
    write_json(args.outdir/'openbabel_verdicts.json', scored)
    print({k: scored[k] for k in ('version', 'n', 'strict', 'unified_resonance', 'smiles_species', 'typed_recall', 'failures')})


if __name__ == '__main__':
    main()
