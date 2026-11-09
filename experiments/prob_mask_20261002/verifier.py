"""A learned verifier over whole candidate graphs, and conformal candidate sets.

Why: on the 488 PDBBind poses the local-search neighbourhood contains the
reference B for 86.7% of molecules, but selecting by the model's own factorized
score lands on it for 67.8%. The remaining ~19 points are a *ranking* problem,
not a search problem, so this trains a small reference-blind scorer over complete
(B,h,q) candidates and then calibrates it into prediction sets.

Data: the 4500 joint training molecules (``joint_v1/sources.pt``), candidates
generated at sigma in {0, 0.1, 0.2} with the same neighbourhood as deployment.
Molecules are split by connectivity skeleton into verifier-train / conformal-
calibration, so a skeleton never appears in both.

Subcommands
-----------
  build   generate and cache the candidate dataset (train molecules)
  train   fit the verifier, report ranking accuracy on the held-out calibration
          molecules, and derive split-conformal thresholds
  apply   score the 488-pose candidates: selection accuracy and set coverage
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as p
import joint as J
import jointt as T
import localsearch as L

OUT = HERE/'joint_pose'


def common(a):
    backbone, bargs = p.load_backbone(a.checkpoint, a.device)
    # load_joint picks JointNet or the t-conditioned JointNetT from the
    # checkpoint's recorded arch, so both are evaluated by this same code.
    net, ck = T.load_joint(a.joint_ckpt, a.device)
    catalogue = J.make_catalogue(torch.load(a.sources, map_location='cpu',
                                           weights_only=False)['train'])
    return backbone, bargs, net, catalogue


def featurize_rows(backbone, bargs, net, catalogue, rows, a, sigmas, tag):
    """Generate labelled candidates for a list of joint rows."""
    data = []
    t0 = time.perf_counter()
    for si, sigma in enumerate(sigmas):
        rng = np.random.default_rng(1000+si)
        for start in range(0, len(rows), a.batch):
            chunk = [p.geometry(r, sigma, rng, bargs['cutoff'])
                     for r in rows[start:start+a.batch]]
            feats = J.extract(backbone, chunk, a.device)
            for r in feats:
                cands, info = L.propose(net, r, catalogue, a.device,
                                        fraction=a.fraction, max_flagged=a.max_flagged,
                                        topk_per_edge=a.topk_per_edge, beam=a.beam)
                L.label_candidates(r, cands)
                data.append({
                    'id': r['id'], 'skeleton': r.get('skeleton', str(r['id'])),
                    'sigma': sigma, 'missed': int(r['missed']),
                    'X': np.stack([c['features'] for c in cands]),
                    'y': np.array([c['is_reference_species'] for c in cands], dtype=np.float32),
                    'yB': np.array([c['is_reference_B'] for c in cands], dtype=np.float32),
                    'model_score': np.array([c['model_score'] for c in cands], dtype=np.float32),
                    'valid': np.array([c['valid'] for c in cands], dtype=np.float32),
                })
            if (start//a.batch) % 10 == 0:
                p.log(f'{tag} sigma={sigma}: {start+len(chunk)}/{len(rows)} '
                      f'({time.perf_counter()-t0:.0f}s)')
    return data


def cmd_build(a):
    RDLogger.DisableLog('rdApp.*'); p.seed_all(11)
    backbone, bargs, net, catalogue = common(a)
    rows = torch.load(a.sources, map_location='cpu', weights_only=False)['train']
    if a.limit:
        rows = rows[:a.limit]
    # joint rows carry 'truth' but not the geometry-dependent 'labels'/'missed';
    # p.geometry adds them, and J.extract adds the frozen features.
    data = featurize_rows(backbone, bargs, net, catalogue, rows, a,
                          a.sigmas, 'train-mols')
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({'data': data, 'feature_names': L.FEATURE_NAMES, 'args': vars(a)},
               OUT/'verifier_dataset.pt')
    n = len(data)
    oracle = np.mean([d['y'].max() for d in data])
    greedy = np.mean([d['y'][0] for d in data])
    best_score = np.mean([d['y'][d['model_score'].argmax()] for d in data])
    p.log(f'built {n} molecule-instances; greedy species {greedy:.4f}, '
          f'model-score selection {best_score:.4f}, neighbourhood oracle {oracle:.4f}')


class Verifier(nn.Module):
    def __init__(self, d, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, width), nn.SiLU(),
                                 nn.Linear(width, width), nn.SiLU(),
                                 nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def target_vector(d, target):
    if target == 'species':
        return d['y']
    if target == 'B':
        return d['yB']
    return (d['y']*d['yB']).astype('float32')


def split_by_skeleton(data, calib_frac, seed=11):
    skeletons = sorted({d['skeleton'] for d in data})
    rng = np.random.default_rng(seed)
    rng.shuffle(skeletons)
    k = int(round(calib_frac*len(skeletons)))
    calib = set(skeletons[:k])
    return ([d for d in data if d['skeleton'] not in calib],
            [d for d in data if d['skeleton'] in calib])


def standardize(train):
    X = np.concatenate([d['X'] for d in train])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def listwise_loss(scores, y):
    """Softmax cross-entropy over a molecule's candidates (listwise ranking)."""
    if y.sum() == 0:
        return None
    target = y/y.sum()
    return -(target*scores.log_softmax(-1)).sum()


def cmd_train(a):
    p.seed_all(11)
    blob = torch.load(OUT/'verifier_dataset.pt', map_location='cpu', weights_only=False)
    data = blob['data']
    train, calib = split_by_skeleton(data, a.calib_frac)
    p.log(f'verifier train instances {len(train)}, calibration {len(calib)}')
    mu, sd = standardize(train)
    model = Verifier(train[0]['X'].shape[1], a.width)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    order = np.arange(len(train))
    for epoch in range(1, a.epochs+1):
        np.random.shuffle(order)
        model.train(); total = count = 0
        for idx in order:
            d = train[idx]
            x = torch.tensor((d['X']-mu)/sd)
            y = torch.tensor(target_vector(d, a.target))
            s = model(x)
            loss = listwise_loss(s, y)
            if loss is None:
                continue
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss); count += 1
        model.eval()
        with torch.no_grad():
            sel = np.mean([target_vector(d, a.target)[int(model(torch.tensor((d['X']-mu)/sd)).argmax())]
                           for d in calib])
        p.log(f'epoch {epoch}: loss {total/max(count,1):.4f}  calib top-1 species {sel:.4f}')
    with torch.no_grad():
        base_greedy = np.mean([target_vector(d, a.target)[0] for d in calib])
        base_score = np.mean([target_vector(d, a.target)[d['model_score'].argmax()] for d in calib])
        oracle = np.mean([target_vector(d, a.target).max() for d in calib])
        probs = []
        for d in calib:
            s = model(torch.tensor((d['X']-mu)/sd)).softmax(-1).numpy()
            tv = target_vector(d, a.target)
            hit = tv.max() > 0
            # conformal score: 1 - (probability mass of the reference candidate)
            probs.append(1.0-float(s[tv > 0].max()) if hit else 1.0)
    probs = np.array(probs)
    qs = {str(al): float(np.quantile(probs, min(1.0, (1-al)*(1+1/len(probs)))))
          for al in a.alphas}
    torch.save({'state': model.state_dict(), 'mu': mu, 'sd': sd,
                'width': a.width, 'thresholds': qs, 'target': a.target,
                'feature_names': blob['feature_names']},
               OUT/f'verifier_{a.target}.pt')
    summary = {'target': a.target,
               'train_instances': len(train), 'calib_instances': len(calib),
               'calib_greedy_species': float(base_greedy),
               'calib_model_score_species': float(base_score),
               'calib_verifier_species': float(sel),
               'calib_neighbourhood_oracle': float(oracle),
               'conformal_thresholds': qs}
    (OUT/f'verifier_train_{a.target}.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    p.log(json.dumps(summary, indent=1))


def cmd_apply(a):
    RDLogger.DisableLog('rdApp.*'); p.seed_all(11)
    backbone, bargs, net, catalogue = common(a)
    ck = torch.load(OUT/f'verifier_{a.target}.pt', map_location='cpu', weights_only=False)
    model = Verifier(len(ck['mu']), ck['width'])
    model.load_state_dict(ck['state']); model.eval()
    mu, sd = ck['mu'], ck['sd']

    wanted = {r['name'] for r in json.loads((OUT/'joint_pose_sigma0_summary.json')
                                            .read_text(encoding='utf-8'))['records']}
    rows = []
    for mol in Chem.SDMolSupplier(str(ROOT/'data/pdbbind_ccd_pose500.sdf'), removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        if name not in wanted:
            continue
        try:
            rows.append(J.make_row(mol, name, -1, 'pdbbind_pose'))
        except Exception:
            continue
        if a.limit and len(rows) >= a.limit:
            break
    rng = np.random.default_rng(11)
    records = []
    with torch.no_grad():
        for start in range(0, len(rows), a.batch):
            chunk = [p.geometry(r, a.sigma, rng, bargs['cutoff'])
                     for r in rows[start:start+a.batch]]
            feats = J.extract(backbone, chunk, a.device)
            for r in feats:
                cands, info = L.propose(net, r, catalogue, a.device,
                                        fraction=a.fraction, max_flagged=a.max_flagged,
                                        topk_per_edge=a.topk_per_edge, beam=a.beam)
                L.label_candidates(r, cands)
                X = np.stack([c['features'] for c in cands])
                s = model(torch.tensor((X-mu)/sd))
                prob = s.softmax(-1).numpy()
                ms = np.array([c['model_score'] for c in cands])
                y = np.array([c['is_reference_species'] for c in cands])
                yB = np.array([c['is_reference_B'] for c in cands])
                # model-score selection with buildability as a hard filter,
                # exactly as pose_localmap.py selects
                valid = np.array([c['valid'] for c in cands])
                order = np.lexsort((ms, valid))
                base_pick = int(order[-1])
                ver_pick = int(prob.argmax())
                rec = {'name': r['id'], 'candidates': len(cands),
                       'greedy_species': bool(y[0]), 'greedy_B': bool(yB[0]),
                       'score_species': bool(y[base_pick]), 'score_B': bool(yB[base_pick]),
                       'verifier_species': bool(y[ver_pick]), 'verifier_B': bool(yB[ver_pick]),
                       'oracle_species': bool(y.max()), 'oracle_B': bool(yB.max()),
                       'verifier_labels': [int(x) for x in cands[ver_pick]['B'].tolist()],
                       'pairs': [list(map(int, e)) for e in r['pairs'].tolist()]}
                tv = (y if a.target == 'species' else
                      yB if a.target == 'B' else (y & yB))
                for al, q in ck['thresholds'].items():
                    keep = prob >= (1.0-q)
                    if not keep.any():
                        keep = prob >= prob.max()
                    rec[f'set_size_{al}'] = int(keep.sum())
                    rec[f'set_cover_species_{al}'] = bool(y[keep].max() if keep.any() else False)
                    rec[f'set_cover_target_{al}'] = bool(tv[keep].max() if keep.any() else False)
                records.append(rec)
            p.log(f'applied {min(start+a.batch, len(rows))}/{len(rows)}')
    n = len(records)
    summary = {'n': n, 'mean_candidates': sum(r['candidates'] for r in records)/n}
    for key in ['greedy', 'score', 'verifier', 'oracle']:
        summary[f'{key}_species'] = sum(r[f'{key}_species'] for r in records)/n
        summary[f'{key}_B'] = sum(r[f'{key}_B'] for r in records)/n
    summary['target'] = a.target
    for al in ck['thresholds']:
        summary[f'conformal_{al}_mean_set_size'] = sum(r[f'set_size_{al}'] for r in records)/n
        summary[f'conformal_{al}_coverage_target'] = sum(r[f'set_cover_target_{al}'] for r in records)/n
        summary[f'conformal_{al}_coverage_species'] = sum(r[f'set_cover_species_{al}'] for r in records)/n
        summary[f'conformal_{al}_nominal_coverage'] = 1-float(al)
    (OUT/f'verifier_pose_{a.target}.json').write_text(json.dumps(
        {'summary': summary, 'thresholds': ck['thresholds'], 'records': records},
        indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('command', choices=['build', 'train', 'apply'])
    a.add_argument('--checkpoint', default=str(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    a.add_argument('--joint_ckpt', default=str(HERE/'joint_v1/best.pt'))
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--sigma', type=float, default=0.0)
    a.add_argument('--sigmas', type=float, nargs='+', default=[0.0, 0.1, 0.2])
    a.add_argument('--fraction', type=float, default=0.10)
    a.add_argument('--max_flagged', type=int, default=8)
    a.add_argument('--topk_per_edge', type=int, default=2)
    a.add_argument('--beam', type=int, default=48)
    a.add_argument('--calib_frac', type=float, default=0.25)
    a.add_argument('--epochs', type=int, default=8)
    a.add_argument('--lr', type=float, default=1e-3)
    a.add_argument('--width', type=int, default=64)
    a.add_argument('--alphas', type=float, nargs='+', default=[0.2, 0.1, 0.05])
    a.add_argument('--target', choices=['species', 'B', 'both'], default='species',
                   help="what the verifier is trained to rank first: canonical-species "
                        "identity, the file's exact B labelling, or both at once")
    a.add_argument('--limit', type=int, default=0)
    args = a.parse_args()
    {'build': cmd_build, 'train': cmd_train, 'apply': cmd_apply}[args.command](args)


if __name__ == '__main__':
    main()
