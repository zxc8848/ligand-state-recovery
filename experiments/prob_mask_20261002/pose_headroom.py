"""Headroom of a *coupled* second pass, measured rather than assumed.

Two ceilings on the 488 PDBBind poses, same frozen encoder / JointNet / catalogue
as the deployed configuration:

  containment  flag the lowest-confidence f fraction of each molecule's candidate
               edges after the one-shot pass; how often is the molecule's entire
               set of wrong edges inside the flagged set? That is the recall
               ceiling of any refinement that only revisits flagged edges, and it
               is a per-molecule number, unlike per-edge error recall.
  true-B       give the reference B and let the pipeline predict (h,q) with the
               catalogue prior and the charge DP; the resulting species rate is
               the ceiling a perfect ranker over B could reach on this benchmark.

Also reports how many wrong edges a molecule has, since a one-edge-at-a-time
refinement cannot repair a coupled 2-swap (which oxygen carries the double bond).
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
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--outdir', default=str(HERE/'joint_pose'))
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--sigma', type=float, default=0.0)
    a.add_argument('--fractions', type=float, nargs='+', default=[0.05, 0.10, 0.15, 0.25])
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
    catalogue = J.make_catalogue(torch.load(a.sources, map_location='cpu', weights_only=False)['train'])

    contained = {f: 0 for f in a.fractions}
    flagged_edges = {f: 0 for f in a.fractions}
    nwrong = Counter()
    true_b_species = true_b_valid = 0
    n = 0
    draws = np.random.default_rng(11)
    with torch.no_grad():
        for i in range(0, len(feats), a.batch):
            chunk = feats[i:i+a.batch]
            b = J.pack(chunk, a.device)
            allmaskB = torch.full_like(b['labels'], J.BMASK)
            allmaskA = torch.full_like(b['atarget'], J.AMASK)
            bl, _ = net(b, allmaskB, allmaskA)
            bp = bl.softmax(-1)
            pred = bp.argmax(-1)
            conf = bp.gather(1, pred[:, None]).squeeze(1)
            wrong = (pred != b['labels'])
            # true-B oracle: reference B visible, (h,q) predicted as deployed
            _, al = net(b, b['labels'], allmaskA)
            al = al + J.catalogue_logits(b, b['labels'], catalogue)
            ad = J.constrained_atoms(al, allmaskA, b['atom_lengths'], b['Q'], draws, greedy=True)
            asplit = ad.cpu().split(b['atom_lengths'])
            start = 0
            for g, r in enumerate(chunk):
                m = b['lengths'][g]
                w = wrong[start:start+m]
                c = conf[start:start+m]
                nwrong[int(w.sum())] += 1
                for f in a.fractions:
                    k = max(1, int(round(f*m)))
                    idx = torch.topk(-c, k).indices
                    # conf/wrong live on the compute device; the flag mask must too,
                    # or indexing it with a CUDA index tensor fails (CPU-only runs
                    # never hit this).
                    flag = torch.zeros(m, dtype=torch.bool, device=c.device)
                    flag[idx] = True
                    flagged_edges[f] += k
                    if bool((w & ~flag).sum() == 0):
                        contained[f] += 1
                smi, good = J.build_species(r, r['labels'], asplit[g])
                true_b_species += bool(smi == r['smiles'])
                true_b_valid += bool(good)
                n += 1
                start += m

    res = {'n_molecules': n,
           'wrong_edge_histogram': dict(sorted(nwrong.items())),
           'molecules_with_zero_wrong_edges': nwrong[0],
           'containment_of_all_wrong_edges': {f'{int(f*100)}%': contained[f]/n for f in a.fractions},
           'mean_flagged_edges_per_molecule': {f'{int(f*100)}%': flagged_edges[f]/n for f in a.fractions},
           'true_B_oracle_species': true_b_species/n,
           'true_B_oracle_valid': true_b_valid/n}
    print(json.dumps(res, indent=1))
    Path(a.outdir, 'pose_headroom.json').write_text(json.dumps(res, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
