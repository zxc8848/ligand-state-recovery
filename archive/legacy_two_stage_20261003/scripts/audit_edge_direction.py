"""Audit frozen heavy-only checkpoints under canonical and legacy vector signs.

This does not retrain, overwrite checkpoints, or replace paper results. The paper's
coordinate inference already uses canonical geometry; the legacy arm simulates the
negated heavy-heavy vectors stored by the old radius graph. Refiner geometry is
reconstructed in both arms, as in the existing training/inference bridges.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bondnet.integration.molcraft import (
    all_atom_candidate_edges, make_coordinate_batch,
    build_refiner_inference_batch, forward_coordinate_refiner,
)
from scripts.evaluate_structured_refiner import load_refiner
from scripts.train_bond_refiner import load_stage1


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference_sdf', default='data/pdbbind_ccd_pose500.sdf')
    parser.add_argument('--stage1_ckpt', default='checkpoints/diverse80k_heavy_stage1/best.pt')
    parser.add_argument('--refiner_ckpt', default='checkpoints/diverse80k_heavy_struct6_vector_hardneg_to50/best.pt')
    parser.add_argument('--noise', nargs='+', type=float, default=[0.0, 0.05, 0.10, 0.15])
    parser.add_argument('--max_mols', type=int, default=None)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/edge_direction_audit_2026-10-02.json')
    args = parser.parse_args()
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    device = torch.device(args.device)
    stage1 = load_stage1(args.stage1_ckpt, device)
    refiner, train_args = load_refiner(args.refiner_ckpt, device)
    allowed = {6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83}
    result = {
        'stage1_ckpt': args.stage1_ckpt, 'refiner_ckpt': args.refiner_ckpt,
        'reference_sdf': args.reference_sdf, 'single_stage': bool(getattr(train_args, 'single_stage', False)),
        'note': 'Sign sensitivity of frozen checkpoints, not a retraining experiment. '
                'Paper coordinate inference already used sender-minus-receiver vectors.',
        'conditions': {},
    }
    for sigma in args.noise:
        cell = dict(n=0, edges=0, stage1_max_logit_delta=0., stage1_binary_changes=0,
                    stage1_probability_max_delta=0., message_graph_changed_molecules=0,
                    refiner_max_logit_delta=0., refiner_argmax_changes=0,
                    refiner_changed_molecules=0)
        for source_index, mol in enumerate(Chem.SDMolSupplier(args.reference_sdf, removeHs=False)):
            if mol is None or mol.GetNumConformers() == 0:
                continue
            mol = Chem.RemoveHs(mol)
            zs = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            if not set(zs) <= allowed:
                continue
            if args.max_mols is not None and cell['n'] >= args.max_mols:
                break
            elems = torch.tensor(zs, device=device)
            coord = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
            if sigma:
                generator = torch.Generator().manual_seed(42 + source_index)
                coord += torch.randn(coord.shape, generator=generator) * sigma
            coord = coord.to(device)
            edges = all_atom_candidate_edges(elems, coord, 3.0, 12, 15, 1.4)
            batch = make_coordinate_batch(elems, coord, Chem.GetFormalCharge(mol),
                                          candidate_edge_index=edges)
            legacy = dict(batch, edge_diff=-batch['edge_diff'])
            canonical_built = build_refiner_inference_batch(stage1, batch, train_args)
            legacy_built = build_refiner_inference_batch(stage1, legacy, train_args)
            canonical_logits = forward_coordinate_refiner(refiner, batch, canonical_built, return_aux=False)
            legacy_logits = forward_coordinate_refiner(refiner, legacy, legacy_built, return_aux=False)
            conn_a, conn_b = canonical_built['all_conn_logits'], legacy_built['all_conn_logits']
            cell['n'] += 1
            cell['edges'] += int(edges.shape[0])
            if conn_a.numel():
                cell['stage1_max_logit_delta'] = max(cell['stage1_max_logit_delta'], float((conn_a-conn_b).abs().max()))
                cell['stage1_probability_max_delta'] = max(cell['stage1_probability_max_delta'], float((conn_a.sigmoid()-conn_b.sigmoid()).abs().max()))
                cell['stage1_binary_changes'] += int(((conn_a >= 0) != (conn_b >= 0)).sum())
            cell['message_graph_changed_molecules'] += int(not torch.equal(canonical_built['message_ei'], legacy_built['message_ei']))
            if canonical_logits.numel():
                delta = float((canonical_logits-legacy_logits).abs().max())
                cell['refiner_max_logit_delta'] = max(cell['refiner_max_logit_delta'], delta)
                cell['refiner_changed_molecules'] += int(delta > 1e-6)
                cell['refiner_argmax_changes'] += int((canonical_logits.argmax(-1) != legacy_logits.argmax(-1)).sum())
        result['conditions'][str(sigma)] = cell
        print(sigma, cell, flush=True)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
