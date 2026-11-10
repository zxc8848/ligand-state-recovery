"""Use the authors' 3D checkpoint and inference path on Z/R-only pose inputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDLogger
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
from evaluation_common import reference_population, score_dump


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'input', 'sdf', 'outdir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--expected-n', type=int, default=491)
    args = p.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import yuel_bond as Y
    torch.set_num_threads(2)
    RDLogger.DisableLog('rdApp.*')
    inputs = json.loads(args.input.read_text(encoding='utf-8'))
    refs, skipped = reference_population(args.sdf)
    if skipped or len(inputs) != args.expected_n or len(refs) != args.expected_n or {r['id'] for r in inputs} != set(refs):
        raise ValueError('Unexpected inference/score denominator')
    args.outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = Y.YuelBond.load_from_checkpoint(str(args.checkpoint), map_location=device).eval().to(device)
    periodic = Chem.GetPeriodicTable()
    label_map = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}
    failures, predicted = {}, {}
    for i, row in enumerate(inputs):
        name, z = row['id'], row['Z']
        symbols = [periodic.GetElementSymbol(x) for x in z]
        unknown = sorted(set(symbols)-set(Y.const.ATOM2IDX))
        if unknown:
            failures[name] = 'unsupported_elements:'+','.join(unknown)
            continue
        atom_types = np.asarray([Y.const.ATOM2IDX[s] for s in symbols])
        raw_data = [dict(name=name, positions=np.asarray(row['R']),
                         atoms=np.eye(Y.const.N_ATOM_TYPES)[atom_types], bonds=np.array([]))]
        try:
            ds = Y.BondDataset(raw_data=raw_data, device=device, has_bonds=False, progress_bar=False)
            batch = next(iter(Y.get_dataloader(ds, batch_size=1, collate_fn=Y.collate)))
            with torch.inference_mode():
                pred = Y.create_molecule_from_predictions(batch['positions'], batch['one_hot'],
                          batch['edge_index'], model.forward(batch), batch['node_mask'], batch['edge_mask'], name=name)
            if [a.GetAtomicNum() for a in pred.GetAtoms()] != z:
                raise ValueError('Changed atom order or inventory')
            unsupported = sorted({str(b.GetBondType()) for b in pred.GetBonds() if b.GetBondType() not in label_map})
            if unsupported:
                failures[name] = 'unsupported_bond_labels:'+','.join(unsupported)
                continue
            predicted[name] = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): label_map[b.GetBondType()]
                               for b in pred.GetBonds()}
        except (ValueError, RuntimeError, IndexError, KeyError) as exc:
            failures[name] = f'{type(exc).__name__}: {exc}'
        if (i+1) % 100 == 0:
            print(f'{i+1}/{len(inputs)} processed', flush=True)
    if not predicted or set(predicted) | set(failures) != set(refs):
        raise RuntimeError('Incomplete predictions or no usable output')
    edge_path = args.outdir/'yuelbond_edges.jsonl'
    with edge_path.open('w', encoding='utf-8') as out:
        for name, graph in predicted.items():
            truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                     4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in refs[name].GetBonds()}
            for i, j in sorted(graph.keys() | truth.keys()):
                out.write(json.dumps(dict(name=name, i=i, j=j, true=truth.get((i,j), 0), pred=graph.get((i,j), 0)))+'\n')
    result = score_dump(edge_path, args.sdf, expected_n=args.expected_n)
    result.update(method='YuelBond 3D published checkpoint; no input H, bonds, or Q',
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  failure_counts=dict(Counter(x.split(':',1)[0] for x in failures.values())), failures=failures)
    (args.outdir/'yuelbond_verdicts.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print({k: result[k] for k in ('n','strict','unified_resonance','smiles_species','failure_counts')})


if __name__ == '__main__':
    main()
