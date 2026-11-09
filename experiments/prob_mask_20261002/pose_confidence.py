"""Is the residual bond-type error a low-confidence error?

Iterative / diffusion-style decoding can only help if the marginals it reveals and
re-masks are informative about where the mistakes are. This measures, on the 488
PDBBind poses, the one-shot all-MASK B marginal at every reference edge and splits
it by whether the argmax label is right, so the usable headroom of any
confidence-ordered schedule is explicit rather than assumed.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import probe as p
import joint as J
import jointt as T


def parse_args():
    a = argparse.ArgumentParser()
    a.add_argument('--sdf', default=str(ROOT/'data/pdbbind_ccd_pose500.sdf'))
    a.add_argument('--names', default=str(HERE/'joint_pose/joint_pose_sigma0_summary.json'))
    a.add_argument('--checkpoint', default=str(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    a.add_argument('--joint_ckpt', default=str(HERE/'joint_v1/best.pt'))
    a.add_argument('--outdir', default=str(HERE/'joint_pose'))
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--sigma', type=float, default=0.0)
    return a.parse_args()


def main():
    a = parse_args()
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(11)
    wanted = {r['name'] for r in json.loads(Path(a.names).read_text(encoding='utf-8'))['records']}
    rows = []
    for mol in Chem.SDMolSupplier(a.sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        if name not in wanted:
            continue
        try:
            rows.append(J.make_row(mol, name, -1, 'pdbbind_pose'))
        except Exception:
            continue
    backbone, bargs = p.load_backbone(a.checkpoint, a.device)
    rng = np.random.default_rng(11)
    geos = [p.geometry(r, a.sigma, rng, bargs['cutoff']) for r in rows]
    feats = []
    for i in range(0, len(geos), a.batch):
        feats.extend(J.extract(backbone, geos[i:i+a.batch], a.device))
    # load_joint picks JointNet or the t-conditioned JointNetT from the
    # checkpoint's recorded arch, so both are evaluated by this same code.
    net, ck = T.load_joint(a.joint_ckpt, a.device)

    conf_ok, conf_bad = [], []
    bad_true = Counter(); bad_pred = Counter()
    mol_entropy = []
    with torch.no_grad():
        for i in range(0, len(feats), a.batch):
            chunk = feats[i:i+a.batch]
            b = J.pack(chunk, a.device)
            bl, _ = net(b, torch.full_like(b['labels'], J.BMASK),
                        torch.full_like(b['atarget'], J.AMASK))
            bp = bl.softmax(-1)
            pred = bp.argmax(-1)
            conf = bp.gather(1, pred[:, None]).squeeze(1)
            true = b['labels']
            ok = pred == true
            conf_ok.extend(conf[ok].tolist())
            conf_bad.extend(conf[~ok].tolist())
            for t, q in zip(true[~ok].tolist(), pred[~ok].tolist()):
                bad_true[t] += 1; bad_pred[q] += 1
            ent = -(bp.clamp_min(1e-12).log()*bp).sum(-1)
            start = 0
            for n in b['lengths']:
                mol_entropy.append(float(ent[start:start+n].sum()))
                start += n

    ok = np.array(conf_ok); bad = np.array(conf_bad)
    total = len(ok)+len(bad)
    res = {
        'reference_edges': total,
        'mistyped_edges': int(len(bad)),
        'edge_accuracy': float(len(ok)/total),
        'confidence_correct': {'mean': float(ok.mean()), 'median': float(np.median(ok)),
                               'frac_gt_0.9': float((ok > .9).mean()),
                               'frac_gt_0.99': float((ok > .99).mean())},
        'confidence_mistyped': {'mean': float(bad.mean()), 'median': float(np.median(bad)),
                                'frac_gt_0.5': float((bad > .5).mean()),
                                'frac_gt_0.9': float((bad > .9).mean()),
                                'frac_gt_0.99': float((bad > .99).mean())},
        'mistyped_by_true_class': dict(sorted(bad_true.items())),
        'mistyped_by_pred_class': dict(sorted(bad_pred.items())),
        'molecules_with_total_entropy_below_0.1': int(sum(1 for e in mol_entropy if e < .1)),
        'n_molecules': len(mol_entropy),
    }
    # how many of the mistyped edges would a top-f% lowest-confidence remask reach
    thresholds = {}
    allconf = np.concatenate([ok, bad])
    labels = np.concatenate([np.zeros(len(ok)), np.ones(len(bad))])
    order = np.argsort(allconf)
    for f in (.05, .10, .15, .25, .50):
        k = int(round(f*total))
        thresholds[f'{int(f*100)}%'] = float(labels[order[:k]].sum()/len(bad))
    res['recall_of_mistyped_edges_by_lowest_confidence_remask'] = thresholds
    print(json.dumps(res, indent=1))
    Path(a.outdir, 'pose_confidence.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
