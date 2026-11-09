"""Evaluate the single-stage + JointNet(B->h,q) + catalogue + Q-DP pipeline on the
488-molecule PDBBind pose benchmark of the template-free paper.

Scoring is delegated to the archived method-neutral scorer by emitting a per-edge
JSONL in exactly the format that scorer consumes, so the new decoder and the
archived two-stage+MILP decoder are compared with identical reference graphs,
identical molecule set and identical build path.

Two dumps are written:
  *_refedges.jsonl   rows restricted to the reference-edge set of the archived
                     dump -> directly comparable with the published numbers
                     (that convention cannot see false-positive edges);
  *_allpairs.jsonl   every 2.5 A candidate pair, so extra predicted bonds are
                     penalised as well.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as p
import joint as J
import jointt as T


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--sdf', default=str(HERE.parents[1]/'data/pdbbind_ccd_pose500.sdf'))
    a.add_argument('--baseline_per_edge', default=str(HERE.parents[1]/'results/per_edge_training_catalogue.jsonl'))
    a.add_argument('--checkpoint', default=str(HERE.parents[1]/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    a.add_argument('--joint_ckpt', default=str(HERE/'joint_v1/best.pt'))
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--outdir', default=str(HERE.parents[1]/'out'))
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--candidates', type=int, default=8)
    a.add_argument('--sigma', type=float, default=0.0)
    a.add_argument('--seeds', type=int, nargs='+', default=[20261002, 20261003, 20261004])
    a.add_argument('--limit', type=int, default=0)
    return a.parse_args()


def main():
    a = parse_args()
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(11)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    # ---- molecule set: exactly the names the archived dump scored ----
    base_rows = defaultdict(list)
    for line in Path(a.baseline_per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            base_rows[r['name']].append(r)
    wanted = set(base_rows)
    p.log(f'archived dump: {len(wanted)} molecules, {sum(len(v) for v in base_rows.values())} rows')

    rows, failures = [], defaultdict(int)
    for mol in Chem.SDMolSupplier(a.sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        if name not in wanted:
            continue
        try:
            row = J.make_row(mol, name, -1, 'pdbbind_pose')
        except Exception as exc:
            failures[str(exc).split('\n')[0][:60]] += 1
            continue
        rows.append(row)
        if a.limit and len(rows) >= a.limit:
            break
    p.log(f'joint-representable poses: {len(rows)}/{len(wanted)}  failures={dict(failures)}')

    # ---- frozen geometry encoder ----
    backbone, bargs = p.load_backbone(a.checkpoint, a.device)
    rng = np.random.default_rng(11)
    geos = [p.geometry(r, a.sigma, rng, bargs['cutoff']) for r in rows]
    feats = []
    t0 = time.perf_counter()
    for i in range(0, len(geos), a.batch):
        feats.extend(J.extract(backbone, geos[i:i+a.batch], a.device))
        if i % (a.batch*8) == 0:
            p.log(f'features {i}/{len(geos)}')
    p.log(f'features done in {time.perf_counter()-t0:.1f}s')

    # ---- JointNet + training catalogue ----
    # load_joint picks JointNet or the t-conditioned JointNetT from the
    # checkpoint's recorded arch, so both are evaluated by this same code.
    net, ck = T.load_joint(a.joint_ckpt, a.device)
    catalogue = J.make_catalogue(torch.load(a.sources, map_location='cpu', weights_only=False)['train'])
    p.log(f'joint checkpoint epoch {ck["epoch"]}; catalogue sizes '
          f'{[len(x) for x in catalogue]}')

    # ---- decode ----
    per_mol = {r['id']: {'row': r} for r in feats}
    for i in range(0, len(feats), a.batch):
        chunk = feats[i:i+a.batch]
        b = J.pack(chunk, a.device)
        cached = net(b, torch.full_like(b['labels'], J.BMASK), torch.full_like(b['atarget'], J.AMASK))
        draws = {}
        # top-1: greedy argmax B, greedy constrained (h,q)
        draws['greedy'] = [J.sample(net, b, 1, 7, greedy=True, cached_logits=cached,
                                    catalogue=catalogue, atoms_after_B=True, combine_catalogue=True)]
        for s in a.seeds:
            draws[f'sample{s}'] = [
                J.sample(net, b, 1, s + i*97 + k*1000000, greedy=False, cached_logits=cached,
                         catalogue=catalogue, atoms_after_B=True, combine_catalogue=True)
                for k in range(a.candidates)]
        for tag, lst in draws.items():
            bs = [x.cpu().split(b['lengths']) for x, _ in lst]
            ats = [y.cpu().split(b['atom_lengths']) for _, y in lst]
            for g, r in enumerate(chunk):
                per_mol[r['id']][tag] = [(x[g], y[g]) for x, y in zip(bs, ats)]
        p.log(f'decoded {min(i+a.batch, len(feats))}/{len(feats)}')

    # ---- native metrics + per-edge dumps ----
    ref_lines, all_lines, records = [], [], []
    for r in feats:
        name = r['id']
        entry = per_mol[name]
        pairs = r['pairs'].tolist()
        bd, ad = entry['greedy'][0]
        pred_map = {tuple(sorted(pr)): int(c) for pr, c in zip(pairs, bd.tolist())}
        true_map = r['truth']
        zl = r['z'].tolist()
        # paper2-compatible rows: reference-edge set of the archived dump
        for orow in base_rows[name]:
            e = (orow['i'], orow['j'])
            ref_lines.append(json.dumps({'name': name, 'sigma': a.sigma, 'i': e[0], 'j': e[1],
                                         'z_i': orow['z_i'], 'z_j': orow['z_j'],
                                         'true': orow['true'],
                                         'pred': pred_map.get(tuple(sorted(e)), 0),
                                         'correct': int(pred_map.get(tuple(sorted(e)), 0) == orow['true'])}))
        for e in sorted(set(pred_map) | set(true_map)):
            all_lines.append(json.dumps({'name': name, 'sigma': a.sigma, 'i': e[0], 'j': e[1],
                                         'z_i': zl[e[0]], 'z_j': zl[e[1]],
                                         'true': int(true_map.get(e, 0)),
                                         'pred': pred_map.get(e, 0),
                                         'correct': int(pred_map.get(e, 0) == int(true_map.get(e, 0)))}))
        rec = {'name': name, 'n_atoms': len(zl), 'Q': r['Q'], 'missed_true_pairs': r['missed'],
               'reference_smiles': r['smiles'],
               'pairs': [list(map(int, e)) for e in pairs],
               'greedy_labels': [int(x) for x in bd.tolist()]}
        smi, good = J.build_species(r, bd, ad)
        rec['greedy_species_hit'] = bool(smi == r['smiles'])
        rec['greedy_valid'] = bool(good)
        rec['greedy_smiles'] = smi
        rec['greedy_graph_hit'] = bool(torch.equal(bd, r['labels']) and r['missed'] == 0)
        rec['greedy_joint_hit'] = bool(rec['greedy_graph_hit'] and torch.equal(ad, r['atom_target']))
        for s in a.seeds:
            hits, valid, uniq = [], 0, set()
            for cbd, cad in entry[f'sample{s}']:
                csmi, cgood = J.build_species(r, cbd, cad)
                hits.append(csmi == r['smiles'])
                valid += bool(cgood)
                if cgood:
                    uniq.add(csmi)
            rec[f'cov8_{s}'] = bool(any(hits))
            rec[f'valid_frac_{s}'] = valid/len(entry[f'sample{s}'])
            rec[f'valid_unique_{s}'] = len(uniq)
            rec[f'top1_sample_{s}'] = bool(hits[0])
            rec[f'cand_labels_{s}'] = [[int(x) for x in cbd.tolist()]
                                       for cbd, _ in entry[f'sample{s}']]
        records.append(rec)

    stem = f'joint_pose_sigma{a.sigma:g}'
    (out/f'{stem}_refedges.jsonl').write_text('\n'.join(ref_lines)+'\n', encoding='utf-8')
    (out/f'{stem}_allpairs.jsonl').write_text('\n'.join(all_lines)+'\n', encoding='utf-8')

    n = len(records)
    summary = {
        'n_scored': n, 'n_archived_dump': len(wanted), 'make_row_failures': dict(failures),
        'sigma': a.sigma, 'candidates': a.candidates, 'seeds': a.seeds,
        'greedy_species_top1': sum(r['greedy_species_hit'] for r in records)/n,
        'greedy_strictB_graph': sum(r['greedy_graph_hit'] for r in records)/n,
        'greedy_joint_label': sum(r['greedy_joint_hit'] for r in records)/n,
        'greedy_valid_fraction': sum(r['greedy_valid'] for r in records)/n,
        'molecules_with_missing_true_pair': sum(r['missed_true_pairs'] > 0 for r in records),
        'sampled_top1_species_mean': float(np.mean([[r[f'top1_sample_{s}'] for s in a.seeds] for r in records])),
        'species_coverage8_mean': float(np.mean([[r[f'cov8_{s}'] for s in a.seeds] for r in records])),
        'species_coverage8_per_seed': {str(s): sum(r[f'cov8_{s}'] for r in records)/n for s in a.seeds},
        'valid_fraction_mean': float(np.mean([[r[f'valid_frac_{s}'] for s in a.seeds] for r in records])),
        'valid_unique_mean': float(np.mean([[r[f'valid_unique_{s}'] for s in a.seeds] for r in records])),
    }
    (out/f'{stem}_summary.json').write_text(json.dumps({'args': vars(a), 'summary': summary,
                                                        'records': records}, indent=1), encoding='utf-8')
    p.log(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
