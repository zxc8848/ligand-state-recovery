"""Training-seed replicates of the baseline JointNet recipe.

Every headline so far rests on a single trained network (`joint_v1/best.pt`,
seed 20261002), while the comparison baseline it beats had three. This trains the
*same* recipe at further seeds so the pose endpoints get training-seed error bars
instead of only sampling-seed ones.

The recipe is copied from ``joint.train`` verbatim -- same warm start, corruption
(``probe.masks`` + ``joint.corrupt_states``), 18 epochs, batch 32, lr 2e-4, AdamW
wd 1e-4, grad clip 2, and the same epoch-selection criterion (all-MASK bond CE +
joint-atom CE). Only ``--seed`` and ``--outdir`` differ, and nothing is written
inside ``joint_v1``.

Run seed 20261002 first as a reproducibility check: it should land on epoch 13
with sum CE ~0.11626, which is what ``joint_v1/best.pt`` recorded. If it does not,
the replicates are not comparable to the published baseline and that has to be
reported rather than averaged over.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as p
import joint as J


def train(a):
    out = Path(a.outdir)
    if (out/'best.pt').exists() and not a.force:
        raise SystemExit(f'refusing to overwrite {out/"best.pt"}; pass --force or '
                         f'choose another --outdir')
    out.mkdir(parents=True, exist_ok=True)
    p.seed_all(a.seed)
    data = torch.load(a.features, map_location='cpu', weights_only=False)['rows']
    net = J.JointNet(width=a.width, layers=a.layers).to(a.device)
    if a.warm_start.lower() == 'none':
        p.log('WARNING: no warm start (smoke/debug only)')
    else:
        ck = torch.load(a.warm_start, map_location='cpu', weights_only=False)
        net.edge.load_state_dict(ck['state'], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []
    best = float('inf')
    t0 = time.perf_counter()
    p.log(f'baseline recipe, seed {a.seed}: '
          f'{sum(x.numel() for x in net.parameters()):,} parameters')
    for epoch in range(1, a.epochs+1):
        net.train()
        order = np.random.permutation(len(data['train']))
        loss_total = batches = 0
        for i in range(0, len(order), a.batch):
            b = J.pack([data['train'][j] for j in order[i:i+a.batch]], a.device)
            bs, ats, bsuper, asuper = J.corrupt_states(b)
            bl, al = net(b, bs, ats)
            loss = (nn.functional.cross_entropy(bl[bsuper], b['labels'][bsuper])
                    + nn.functional.cross_entropy(al[asuper], b['atarget'][asuper]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optimizer.step()
            loss_total += loss.item()
            batches += 1
        net.eval()
        vb, va = J.val_ce(net, data['val'], a.device, a.batch)
        row = {'epoch': epoch, 'train_loss': loss_total/batches, 'val_bond_CE': vb,
               'val_atom_joint_CE': va, 'seconds': time.perf_counter()-t0}
        history.append(row)
        p.log(str(row))
        payload = {'state': net.state_dict(), 'args': vars(a), 'epoch': epoch}
        torch.save(payload, out/'last.pt')
        if vb+va < best:
            best = vb+va
            torch.save(payload, out/'best.pt')
        p.save_json(out/'training.json', {
            'history': history,
            'selection': 'all-mask bond CE + joint-atom CE (same as baseline)',
            'parameters': sum(x.numel() for x in net.parameters()),
            'recipe': 'verbatim joint.train; only --seed and --outdir differ',
            'seed': a.seed,
            'reference': 'joint_v1/best.pt was this recipe at seed 20261002, '
                         'epoch 13, sum CE 0.11626'})
    chosen = min(history, key=lambda r: r['val_bond_CE']+r['val_atom_joint_CE'])
    p.log(f'seed {a.seed}: selected epoch {chosen["epoch"]}, sum CE '
          f'{chosen["val_bond_CE"]+chosen["val_atom_joint_CE"]:.5f} '
          f'(joint_v1 reference: epoch 13, 0.11626)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--features', default=str(HERE/'joint_v1/features.pt'))
    ap.add_argument('--warm_start', default=str(HERE/'pilot_v2/decoder_best.pt'))
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--lr', type=float, default=.0002)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.outdir is None:
        a.outdir = str(HERE/f'joint_v5/seed{a.seed}')
    torch.set_num_threads(4)
    train(a)


if __name__ == '__main__':
    main()
