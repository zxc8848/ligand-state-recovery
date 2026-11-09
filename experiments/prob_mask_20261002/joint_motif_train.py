"""Retrain JointNet with motif-level masking and chemistry-aware edge corruption.

Why these two terms, from the measurements in
``results/MASK_REFINEMENT_DESIGN_2026-10-03.md`` on the 488 PDBBind poses:

  * 413 of 427 residual edge errors are single/double confusions, and 112 of the
    176 imperfect molecules have two or more wrong edges at once -- the errors are
    coupled, not independent;
  * re-predicting flagged edges one at a time *lowers* species recovery
    (0.7254 -> 0.7131 over five confidence rounds), while a joint assignment over
    the flagged set raises it to 0.8340.

The baseline corruption (``joint.corrupt_states`` + ``probe.masks``) never asks
the network for a joint distribution over a chemically coupled group, and never
shows it a *plausibly wrong* visible label to repair: edge corruption is uniform
over the five classes and atom corruption is uniform over the 25 states. So the
coupled assignment currently has to be brute-forced at decode time by
``pose_localmap.py``. The two added terms put that work into training:

  motif masking         with probability --p_motif, mask every edge of one or
                        more chemically coupled groups together (terminal
                        heteroatom groups, aromatic systems, double-bond stars)
                        and supervise them jointly;
  chemistry-aware noise with probability --p_chem, flip single<->double on a few
                        *visible* edges, preferring edges inside a terminal
                        heteroatom group, and supervise those edges so the model
                        learns to repair a plausible mistake rather than trust it.

Everything else is identical to the baseline recipe (same warm start, 18 epochs,
batch 32, lr 2e-4, AdamW wd 1e-4, grad clip 2, same epoch-selection criterion:
all-MASK bond CE + joint-atom CE), and the default seed matches, so
``joint_v1/best.pt`` is a matched control that differs only in the corruption.

Writes to its own directory and refuses to overwrite an existing run.
"""
import argparse
import json
import math
import time
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as p
import joint as J

TERMINAL_CENTRES = (6, 7, 15, 16)
TERMINAL_LIGANDS = (8, 16)


def batch_motifs(b):
    """Chemically coupled edge groups per molecule, from the reference graph.

    Returns a list of lists of edge indices (into the flattened batch edge order).
    Molecules are disjoint in the packed node indexing, so components never cross
    molecule boundaries and no per-molecule bookkeeping is needed.
    """
    pairs = b['pairs'].cpu().tolist()
    labels = b['labels'].cpu().tolist()
    z = b['z'].cpu().tolist()
    nnode = b['nnode']

    incident = [[] for _ in range(nnode)]
    degree = [0]*nnode
    for e, ((i, j), c) in enumerate(zip(pairs, labels)):
        if c:
            incident[i].append(e)
            incident[j].append(e)
            degree[i] += 1
            degree[j] += 1

    groups = []

    # 1. terminal heteroatom groups: a centre with >= 2 degree-1 O/S neighbours
    for centre in range(nnode):
        if z[centre] not in TERMINAL_CENTRES or degree[centre] < 2:
            continue
        terminal = [e for e in incident[centre]
                    if (lambda o: z[o] in TERMINAL_LIGANDS and degree[o] == 1)(
                        pairs[e][1] if pairs[e][0] == centre else pairs[e][0])]
        if len(terminal) >= 2:
            groups.append(terminal)

    # 2. aromatic systems: connected components of aromatic (class 4) edges
    arom = [e for e, c in enumerate(labels) if c == 4]
    if arom:
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for e in arom:
            for node in pairs[e]:
                parent.setdefault(node, node)
        for e in arom:
            i, j = pairs[e]
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj
        comp = {}
        for e in arom:
            comp.setdefault(find(pairs[e][0]), []).append(e)
        groups.extend(comp.values())

    # 3. double-bond stars: a double bond plus every edge incident to its endpoints
    for e, c in enumerate(labels):
        if c == 2:
            i, j = pairs[e]
            star = sorted(set(incident[i]) | set(incident[j]))
            if len(star) >= 2:
                groups.append(star)

    return [g for g in groups if g]


def corrupt_states_motif(b, p_motif=0.5, p_chem=0.3, max_motifs=2,
                         chem_edges=2, variant='both'):
    """Baseline corruption plus motif masking and chemistry-aware edge noise.

    The first block is the baseline ``joint.corrupt_states`` verbatim, so the
    control differs from this only by the added terms.
    """
    device = b['labels'].device
    bs, bsuper = p.masks(b)
    ats = b['atarget'].clone()
    amask = torch.zeros_like(ats, dtype=torch.bool)
    start = 0
    for n in b['atom_lengths']:
        rate = .2+.8*torch.rand((), device=device)
        if torch.rand((), device=device) < .3:
            rate = 1.
        amask[start:start+n] = torch.rand(n, device=device) < rate
        start += n
    if torch.rand((), device=device) < .4:
        stars = amask[b['pairs']].any(1)
        bs[stars] = J.BMASK
        bsuper |= stars
    corrupt = (~amask) & (torch.rand(len(ats), device=device) < .03)
    ats[corrupt] = torch.randint(0, 25, (int(corrupt.sum()),), device=device)
    ats[amask] = J.AMASK
    if not amask.any():
        ats[0] = J.AMASK
        amask[0] = True

    stats = {'motif_masked_edges': 0, 'chem_corrupted_edges': 0}
    groups = None

    # --- motif-level masking: one coupled group is masked and supervised as a unit
    if variant in ('motif', 'both') and torch.rand((), device=device) < p_motif:
        groups = batch_motifs(b)
        if groups:
            k = 1+int(torch.randint(0, max_motifs, (1,)).item())
            pick = torch.randperm(len(groups))[:k].tolist()
            ids = sorted({e for g in pick for e in groups[g]})
            index = torch.tensor(ids, device=device, dtype=torch.long)
            bs[index] = J.BMASK
            bsuper[index] = True
            stats['motif_masked_edges'] = len(ids)
            # half the time also hide the states of the atoms the group touches,
            # so the joint (B,h,q) assignment over the motif has to be produced
            if torch.rand((), device=device) < .5:
                nodes = b['pairs'][index].reshape(-1).unique()
                ats[nodes] = J.AMASK
                amask[nodes] = True

    # --- chemistry-aware corruption: a plausible wrong label, supervised
    if variant in ('chem', 'both') and torch.rand((), device=device) < p_chem:
        if groups is None:
            groups = batch_motifs(b)
        visible = (~bsuper) & ((bs == 1) | (bs == 2))
        preferred = torch.zeros_like(visible)
        for g in groups[:64]:
            preferred[torch.tensor(g, device=device, dtype=torch.long)] = True
        pool = (visible & preferred).nonzero().reshape(-1)
        if len(pool) == 0:
            pool = visible.nonzero().reshape(-1)
        if len(pool):
            k = min(chem_edges, len(pool))
            pick = pool[torch.randperm(len(pool), device=device)[:k]]
            bs[pick] = torch.where(bs[pick] == 1,
                                   torch.full_like(bs[pick], 2),
                                   torch.full_like(bs[pick], 1))
            bsuper[pick] = True
            stats['chem_corrupted_edges'] = int(k)

    # Explicit whole-state all-MASK examples match the start of inference.
    if torch.rand((), device=device) < .2:
        ats.fill_(J.AMASK)
        bs.fill_(J.BMASK)
        amask.fill_(True)
        bsuper.fill_(True)
    return bs, ats, bsuper, amask | corrupt, stats


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
        p.log('WARNING: no warm start (smoke/debug only); the baseline run warm '
              'starts net.edge from pilot_v2/decoder_best.pt')
    else:
        ck = torch.load(a.warm_start, map_location='cpu', weights_only=False)
        net.edge.load_state_dict(ck['state'], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []
    best = float('inf')
    t0 = time.perf_counter()
    totals = {'motif_masked_edges': 0, 'chem_corrupted_edges': 0, 'batches': 0}
    p.log(f'motif/chem train parameters: {sum(x.numel() for x in net.parameters()):,} '
          f'variant={a.variant} p_motif={a.p_motif} p_chem={a.p_chem}')
    for epoch in range(1, a.epochs+1):
        net.train()
        order = np.random.permutation(len(data['train']))
        loss_total = batches = 0
        for i in range(0, len(order), a.batch):
            b = J.pack([data['train'][j] for j in order[i:i+a.batch]], a.device)
            bs, ats, bsuper, asuper, stats = corrupt_states_motif(
                b, a.p_motif, a.p_chem, a.max_motifs, a.chem_edges, a.variant)
            bl, al = net(b, bs, ats)
            loss = (nn.functional.cross_entropy(bl[bsuper], b['labels'][bsuper])
                    + nn.functional.cross_entropy(al[asuper], b['atarget'][asuper]))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.)
            optimizer.step()
            loss_total += loss.item()
            batches += 1
            totals['motif_masked_edges'] += stats['motif_masked_edges']
            totals['chem_corrupted_edges'] += stats['chem_corrupted_edges']
            totals['batches'] += 1
        net.eval()
        vb, va = J.val_ce(net, data['val'], a.device, a.batch)
        row = {'epoch': epoch, 'train_loss': loss_total/batches, 'val_bond_CE': vb,
               'val_atom_joint_CE': va, 'seconds': time.perf_counter()-t0}
        history.append(row)
        p.log(str(row))
        payload = {'state': net.state_dict(), 'args': vars(a), 'epoch': epoch}
        # last.pt lets a later ablation check whether the *selection* criterion is
        # the limiting factor: it is the baseline's all-MASK CE, which this run's
        # corruption no longer matches, so the chosen epoch can be pessimistic.
        torch.save(payload, out/'last.pt')
        if vb+va < best:
            best = vb+va
            torch.save(payload, out/'best.pt')
        p.save_json(out/'training.json', {
            'history': history,
            'selection': 'all-mask bond CE + joint-atom CE (same as baseline)',
            'parameters': sum(x.numel() for x in net.parameters()),
            'corruption': {'variant': a.variant, 'p_motif': a.p_motif,
                           'p_chem': a.p_chem, 'max_motifs': a.max_motifs,
                           'chem_edges': a.chem_edges},
            'corruption_totals': totals,
            'control': 'experiments/prob_mask_20261002/joint_v1/best.pt (same seed, '
                       'schedule and warm start; baseline corruption)'})
    p.log(f'best epoch selected; wrote {out/"best.pt"}')


def tests(a):
    """Invariants of the new corruption, on real rows and synthetic features."""
    p.seed_all(7)
    rows = torch.load(a.sources, map_location='cpu', weights_only=False)['train'][:48]
    made = 0
    for r in rows:
        pairs = torch.tensor(sorted(r['truth']), dtype=torch.long)
        labels = torch.tensor([r['truth'][tuple(e.tolist())] for e in pairs],
                              dtype=torch.long)
        row = dict(r, pairs=pairs, labels=labels, missed=0, sigma=0.0,
                   feat=torch.randn(len(pairs), 360).half(),
                   node_feat=torch.randn(len(r['z']), 360).half(),
                   base=torch.randn(len(pairs), 5).log_softmax(-1))
        b = J.pack([row], 'cpu')
        groups = batch_motifs(b)
        assert all(len(g) >= 2 for g in groups), 'motif groups must couple >= 2 edges'
        for _ in range(12):
            bs, ats, bsuper, asuper, stats = corrupt_states_motif(
                b, p_motif=1.0, p_chem=1.0, variant='both')
            assert bs.shape == b['labels'].shape and ats.shape == b['atarget'].shape
            assert bool(((bs == J.BMASK) <= bsuper).all()), \
                'every masked edge must be supervised'
            assert bool(((ats == J.AMASK) <= asuper).all()), \
                'every masked atom must be supervised'
            wrong = (bs != J.BMASK) & (bs != b['labels'])
            assert bool((wrong <= bsuper).all()), \
                'every corrupted visible edge must be supervised'
            assert int(((bs == 1) | (bs == 2) | (bs == 0) | (bs == 3) | (bs == 4)
                        | (bs == J.BMASK)).sum()) == len(bs)
            made += 1
        if groups:
            # a motif mask must be able to hide a whole coupled group at once
            found = False
            for _ in range(40):
                bs, _, bsuper, _, stats = corrupt_states_motif(
                    b, p_motif=1.0, p_chem=0.0, variant='motif')
                if stats['motif_masked_edges'] >= 2:
                    found = True
                    break
            assert found, 'motif masking never fired'
    print(f'ok: {made} corruption draws over {len(rows)} molecules; '
          f'invariants hold (masked and corrupted positions are supervised, '
          f'motif groups couple >= 2 edges)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['train', 'test'])
    ap.add_argument('--features', default=str(HERE/'joint_v1/features.pt'))
    ap.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    ap.add_argument('--warm_start', default=str(HERE/'pilot_v2/decoder_best.pt'),
                    help="edge-subnet warm start, as in the baseline; 'none' disables "
                         "it and is for smoke tests only")
    ap.add_argument('--outdir', default=str(HERE/'joint_v3/motif_train'))
    ap.add_argument('--variant', choices=['motif', 'chem', 'both'], default='both')
    ap.add_argument('--p_motif', type=float, default=0.5)
    ap.add_argument('--p_chem', type=float, default=0.3)
    ap.add_argument('--max_motifs', type=int, default=2)
    ap.add_argument('--chem_edges', type=int, default=2)
    ap.add_argument('--seed', type=int, default=20261002)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--width', type=int, default=96)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--lr', type=float, default=.0002)
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    torch.set_num_threads(4)
    {'train': train, 'test': tests}[a.action](a)


if __name__ == '__main__':
    main()
