"""Retrain the joint (B,h,q) completion network with noise-level (t) conditioning.

Two variants, so the two changes can be attributed separately:

  tcond      the baseline corruption mixture, unchanged (``probe.masks`` +
             ``joint.corrupt_states``), with the t embedding added.
             baseline vs tcond isolates *the t embedding*.
  diffusion  textbook absorbing-state corruption instead: draw a per-molecule
             normalised time, mask that cosine-scheduled fraction of edges and of
             atom states uniformly at random, supervise exactly the masked
             positions. Atom states are fully masked half the time, which is the
             state the deployed B -> (h,q) sampler actually passes in.
             tcond vs diffusion isolates *the explicit schedule*.

Everything else matches the baseline recipe (same warm start, seed 20261002,
18 epochs, batch 32, lr 2e-4, AdamW wd 1e-4, grad clip 2, same epoch-selection
criterion), so ``joint_v1/best.pt`` stays a matched control. ``last.pt`` is saved
every epoch as well, because the selection criterion is the baseline's all-MASK CE
and need not be the best stopping rule for these runs.

Writes to its own directory and refuses to overwrite an existing run.
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
import jointt as T


def corrupt_diffusion(b, p_allmask=0.2, p_atoms_all=0.5):
    """Absorbing-state corruption at a scheduled, per-molecule noise level."""
    device = b['labels'].device
    bs = b['labels'].clone()
    ats = b['atarget'].clone()
    bmask = torch.zeros_like(bs, dtype=torch.bool)
    amask = torch.zeros_like(ats, dtype=torch.bool)

    start = 0
    for n in b['lengths']:
        if n:
            if torch.rand((), device=device) < p_allmask:
                frac = 1.0
            else:
                frac = T.cosine_gamma(float(torch.rand((), device=device)))
            k = int(round(frac*n))
            if k == 0 and n:
                k = 1
            idx = torch.randperm(n, device=device)[:k]
            bmask[start+idx] = True
        start += n

    start = 0
    for n in b['atom_lengths']:
        if n:
            if torch.rand((), device=device) < p_atoms_all:
                frac = 1.0
            else:
                frac = T.cosine_gamma(float(torch.rand((), device=device)))
            k = int(round(frac*n))
            if k == 0 and n:
                k = 1
            idx = torch.randperm(n, device=device)[:k]
            amask[start+idx] = True
        start += n

    bs[bmask] = J.BMASK
    ats[amask] = J.AMASK
    return bs, ats, bmask, amask


def train(a):
    out = Path(a.outdir)
    if (out/'best.pt').exists() and not a.force:
        raise SystemExit(f'refusing to overwrite {out/"best.pt"}; pass --force or '
                         f'choose another --outdir')
    out.mkdir(parents=True, exist_ok=True)
    p.seed_all(a.seed)
    data = torch.load(a.features, map_location='cpu', weights_only=False)['rows']
    net = T.JointNetT(width=a.width, layers=a.layers, n_freq=a.n_freq).to(a.device)
    if a.warm_start.lower() == 'none':
        p.log('WARNING: no warm start (smoke/debug only)')
    else:
        ck = torch.load(a.warm_start, map_location='cpu', weights_only=False)
        net.edge.load_state_dict(ck['state'], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []
    best = float('inf')
    t0 = time.perf_counter()
    p.log(f'{a.variant} parameters: {sum(x.numel() for x in net.parameters()):,} '
          f'(base JointNet is 388,499)')
    for epoch in range(1, a.epochs+1):
        net.train()
        order = np.random.permutation(len(data['train']))
        loss_total = batches = 0
        for i in range(0, len(order), a.batch):
            b = J.pack([data['train'][j] for j in order[i:i+a.batch]], a.device)
            if a.variant == 'diffusion':
                bs, ats, bsuper, asuper = corrupt_diffusion(b, a.p_allmask, a.p_atoms_all)
            else:
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
            'variant': a.variant,
            'arch': 't',
            'control': 'joint_v1/best.pt (no t conditioning, baseline corruption); '
                       'tcond isolates the embedding, diffusion adds the schedule'})
    p.log(f'wrote {out/"best.pt"} and {out/"last.pt"}')


def tests(a):
    p.seed_all(3)
    T.tests()
    rows = torch.load(a.sources, map_location='cpu', weights_only=False)['train'][:16]
    packed = []
    for r in rows:
        pairs = torch.tensor(sorted(r['truth']), dtype=torch.long)
        labels = torch.tensor([r['truth'][tuple(e.tolist())] for e in pairs],
                              dtype=torch.long)
        packed.append(dict(r, pairs=pairs, labels=labels, missed=0, sigma=0.0,
                           feat=torch.randn(len(pairs), 360).half(),
                           node_feat=torch.randn(len(r['z']), 360).half(),
                           base=torch.randn(len(pairs), 5).log_softmax(-1)))
    b = J.pack(packed, 'cpu')
    seen_b, seen_a, draws = [], [], 0
    for _ in range(60):
        bs, ats, bsuper, asuper = corrupt_diffusion(b)
        assert bool(((bs == J.BMASK) == bsuper).all()), \
            'supervised set must be exactly the masked edges'
        assert bool(((ats == J.AMASK) == asuper).all()), \
            'supervised set must be exactly the masked atoms'
        assert bool((bs[~bsuper] == b['labels'][~bsuper]).all()), \
            'unmasked edges must keep the reference label (absorbing corruption only)'
        assert bool((ats[~asuper] == b['atarget'][~asuper]).all())
        tB, tA, _, _ = T.masked_fractions(b, bs, ats)
        seen_b += tB.tolist()
        seen_a += tA.tolist()
        draws += 1
    import numpy as _np
    sb, sa = _np.array(seen_b), _np.array(seen_a)
    assert sb.min() < .3 and sb.max() > .9, f'edge noise levels not spread: {sb.min():.2f}-{sb.max():.2f}'
    assert (sa == 1.).mean() > .3, 'atom states rarely fully masked; the deployed sampler needs that case'
    print(f'ok: {draws} diffusion corruption draws; edge t in '
          f'[{sb.min():.2f},{sb.max():.2f}] mean {sb.mean():.2f}; '
          f'atom t==1 in {(sa == 1.).mean():.0%} of molecules')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['train', 'test'])
    ap.add_argument('--variant', choices=['tcond', 'diffusion'], default='diffusion')
    ap.add_argument('--features', default=str(HERE/'joint_v1/features.pt'))
    ap.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    ap.add_argument('--warm_start', default=str(HERE/'pilot_v2/decoder_best.pt'))
    ap.add_argument('--outdir', default=None)
    ap.add_argument('--p_allmask', type=float, default=0.2)
    ap.add_argument('--p_atoms_all', type=float, default=0.5)
    ap.add_argument('--n_freq', type=int, default=T.N_FREQ)
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--lr', type=float, default=.0002)
    ap.add_argument('--arch', default='t', help='recorded in the checkpoint so the '
                                               'evaluation scripts pick JointNetT')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    if a.outdir is None:
        a.outdir = str(HERE/f'joint_v4/{a.variant}_train')
    torch.set_num_threads(4)
    {'train': train, 'test': tests}[a.action](a)


if __name__ == '__main__':
    main()
