"""Isolated fixed-budget sampling and matched block-training experiments.

Never changes joint_v1. Oracle targets are used ONLY by named diagnostics.
All selected configurations remain developmental, not held-out confirmation.
"""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from rdkit import RDLogger

import joint as j
import jointt as jt
import probe as p

OUT = p.HERE / 'joint_v2'


def load_net(checkpoint, device):
    # jointt.load_joint picks JointNet or the t-conditioned JointNetT from the
    # checkpoint's recorded arch, so both are evaluated by this same code.
    return jt.load_joint(checkpoint, device)


def prior_with_support(b, bd, catalogue):
    exact, coarse, by_z = catalogue
    z = b['z'].cpu().tolist()
    neighbors = [[] for _ in z]
    for (i, k), c in zip(b['pairs'].cpu().tolist(), bd.cpu().tolist()):
        if c:
            neighbors[i].append((z[k], c)); neighbors[k].append((z[i], c))
    reliability = []
    for zi, nei in zip(z, neighbors):
        counts = exact.get((zi, tuple(sorted(nei)))); scale = 1.
        if counts is None:
            counts = coarse.get((zi, tuple(sorted(c for _, c in nei)))); scale = .5
        if counts is None:
            counts = by_z.get(zi, np.zeros(25)); scale = .25
        support = float(counts.sum())
        reliability.append(scale * support / (support + 20.))
    return j.catalogue_logits(b, bd, catalogue), torch.tensor(reliability, device=b['z'].device)[:, None]


@torch.no_grad()
def draws(net, b, catalogue, seed, groups=8, atoms=1, weight=1., adaptive=False,
          anchor=False, oracle=None):
    assert groups * atoms == 8, 'Eight raw proposals, no replenishment'
    bs = torch.full_like(b['labels'], j.BMASK)
    ats = torch.full_like(b['atarget'], j.AMASK)
    bl, _ = net(b, bs, ats)
    bp = bl.softmax(-1)
    output = []
    for group in range(groups):
        gseed = seed + group * 1000000
        gen = torch.Generator(device=bp.device).manual_seed(gseed)
        if oracle == 'B':
            bd = b['labels'].clone()
        elif anchor and group == 0:
            bd = bp.argmax(-1)
        else:
            bd = torch.multinomial(bp, 1, generator=gen).squeeze(1)
        _, al = net(b, bd, ats)
        if weight:
            if adaptive:
                prior, reliability = prior_with_support(b, bd, catalogue)
                al = al + weight * reliability * prior
            else:
                al = al + weight * j.catalogue_logits(b, bd, catalogue)
        for rep in range(atoms):
            # Equal across lambda settings; different draws for each h/q branch.
            rseed = seed + (group * atoms + rep) * 1000000
            if oracle == 'HQ':
                ad = b['atarget'].clone()
            else:
                ad = j.constrained_atoms(al, ats, b['atom_lengths'], b['Q'],
                                         np.random.default_rng(rseed),
                                         greedy=anchor and group == 0 and rep == 0)
            output.append((bd.clone(), ad))
    return output


def configs(names):
    definitions = {
        'w0': dict(weight=0.), 'w025': dict(weight=.25), 'w05': dict(weight=.5),
        'w1': dict(weight=1.), 'w2': dict(weight=2.),
        'adaptive': dict(weight=1., adaptive=True),
        '4x2': dict(groups=4, atoms=2), '2x4': dict(groups=2, atoms=4),
        'anchor': dict(anchor=True),
        'oracle_B': dict(groups=1, atoms=8, oracle='B'),
        'oracle_B_w0': dict(groups=1, atoms=8, oracle='B', weight=0.),
        'oracle_HQ': dict(oracle='HQ'),
        '4x2_w05': dict(groups=4, atoms=2, weight=.5),
        '2x4_w05': dict(groups=2, atoms=4, weight=.5),
        'anchor_w05': dict(anchor=True, weight=.5),
    }
    return [(name, definitions[name]) for name in names]


def score_records(rows, output, name, sigma):
    records = []
    for r, choices in zip(rows, output):
        hits = []; valid = []; species = []; graph = []; joint = []
        for bd, ad in choices:
            smi, good = j.build_species(r, bd, ad)
            hits.append(good and smi == r['smiles']); valid.append(good)
            species.append(smi if good else None)
            exact = r['missed'] == 0 and torch.equal(bd, r['labels'])
            graph.append(exact); joint.append(exact and torch.equal(ad, r['atom_target']))
        records.append(dict(id=r['id'], domain=r['domain'], skeleton=r['skeleton'],
                            Q=r['Q'], has_nH='[nH]' in r['smiles'], method=name, sigma=sigma,
                            species_hits=hits, graph_hits=graph, joint_hits=joint,
                            valid_count=sum(valid), valid_unique=len(set(s for s in species if s)),
                            candidate_species=species, reference_smiles=r['smiles'],
                            missed_true_candidate_edges=r['missed'],
                            Q_preserved=all(int((ad // 5 - 2).sum()) == r['Q'] for _, ad in choices)))
    return records


def summarize(records, seconds, total_n):
    summary = []
    subsets = {'all': records, 'geom_core': [r for r in records if r['domain'] == 'geom_core'],
               'charge_challenge': [r for r in records if r['domain'] == 'diverse_charge'],
               'nH': [r for r in records if r['has_nH']]}
    for name, rr in subsets.items():
        if not rr:
            continue
        summary.append(dict(method=rr[0]['method'], sigma=rr[0]['sigma'], subset=name, n=len(rr),
                            budget=8, species_coverage=np.mean([any(r['species_hits']) for r in rr]),
                            graph_coverage=np.mean([any(r['graph_hits']) for r in rr]),
                            joint_coverage=np.mean([any(r['joint_hits']) for r in rr]),
                            validity=np.mean([r['valid_count'] / 8 for r in rr]),
                            valid_unique_mean=np.mean([r['valid_unique'] for r in rr]),
                            no_valid_fraction=np.mean([r['valid_count'] == 0 for r in rr]),
                            valid_but_miss_fraction=np.mean([r['valid_count'] > 0 and not any(r['species_hits']) for r in rr]),
                            Q_preserved=all(r['Q_preserved'] for r in rr),
                            missing_candidate_molecules=sum(r['missed_true_candidate_edges'] > 0 for r in rr),
                            decoder_ms=seconds / total_n * 1000))
    return summary


def evaluate(a):
    RDLogger.DisableLog('rdApp.*'); p.seed_all(a.seed)
    data = torch.load(j.HERE / 'features.pt', weights_only=False)['rows']
    net, ck = load_net(a.checkpoint, a.device)
    catalogue = j.make_catalogue(torch.load(j.HERE / 'sources.pt', weights_only=False)['train'])
    records = []; summary = []
    for sigma in a.noises:
        rows = [r for r in data['val'] if r['sigma'] == sigma]
        for name, settings in configs(a.configs):
            output = []; seconds = 0.
            for i in range(0, len(rows), a.batch):
                rr = rows[i:i + a.batch]; b = j.pack(rr, a.device)
                p.sync(); start = time.perf_counter()
                choices = draws(net, b, catalogue, a.seed + round(sigma * 10000) + i * 97, **settings)
                p.sync(); seconds += time.perf_counter() - start
                bonds = [bd.cpu().split(b['lengths']) for bd, _ in choices]
                atoms = [ad.cpu().split(b['atom_lengths']) for _, ad in choices]
                output.extend([[(bd[g], ad[g]) for bd, ad in zip(bonds, atoms)] for g in range(len(rr))])
            rr = score_records(rows, output, name, sigma); records.extend(rr)
            ss = summarize(rr, seconds, len(rows)); summary.extend(ss)
            for row in ss:
                if row['subset'] in ['geom_core', 'charge_challenge']:
                    p.log(str({key: row[key] for key in ['method', 'sigma', 'subset', 'species_coverage',
                                                        'validity', 'valid_unique_mean', 'decoder_ms']}))
            p.save_json(OUT / a.output, dict(args=vars(a), checkpoint_epoch=ck['epoch'],
                                           summary=summary, records=records,
                                           scope='Developmental validation; fixed 8 raw proposals; no replenishment. Oracle methods are diagnostics only.'))


@torch.no_grad()
def validation(net, rows, device, batch):
    values = np.zeros(4); counts = np.zeros(2)
    for i in range(0, len(rows), batch):
        b = j.pack(rows[i:i + batch], device)
        ats = torch.full_like(b['atarget'], j.AMASK)
        bl, al = net(b, torch.full_like(b['labels'], j.BMASK), ats)
        _, conditional = net(b, b['labels'], ats)
        values[0] += nn.functional.cross_entropy(bl, b['labels'], reduction='sum').item()
        values[1] += nn.functional.cross_entropy(al, b['atarget'], reduction='sum').item()
        values[2] += nn.functional.cross_entropy(conditional, b['atarget'], reduction='sum').item()
        counts += [len(bl), len(al)]
    return dict(bond_CE=values[0] / counts[0], atom_allmask_CE=values[1] / counts[1],
                atom_conditional_CE=values[2] / counts[1])


def train(a):
    p.seed_all(a.seed)
    data = torch.load(j.HERE / 'features.pt', weights_only=False)['rows']
    net, ck = load_net(a.checkpoint, a.device)
    target = OUT / a.arm; target.mkdir(exist_ok=True)
    if (target / 'best.pt').exists():
        raise FileExistsError(f'Refusing to overwrite training run {target}')
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []; best = float('inf'); start = time.perf_counter()
    for epoch in range(a.epochs + 1):
        loss_sum = batches = 0
        if epoch:
            net.train(); order = np.random.permutation(len(data['train']))
            for i in range(0, len(order), a.batch):
                b = j.pack([data['train'][k] for k in order[i:i + a.batch]], a.device)
                bs, ats, bsuper, asuper = j.corrupt_states(b)
                bl, al = net(b, bs, ats)
                original = nn.functional.cross_entropy(bl[bsuper], b['labels'][bsuper])
                original += nn.functional.cross_entropy(al[asuper], b['atarget'][asuper])
                # Both arms get the SAME extra forward/loss and compute budget.
                # Only its visible B changes: all-MASK control vs true B block.
                condition = b['labels'] if a.arm == 'block_train' else torch.full_like(b['labels'], j.BMASK)
                _, atom_logits = net(b, condition, torch.full_like(b['atarget'], j.AMASK))
                loss = original + .5 * nn.functional.cross_entropy(atom_logits, b['atarget'])
                optimizer.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
                loss_sum += loss.item(); batches += 1
        net.eval(); metrics = validation(net, data['val'], a.device, a.batch)
        selection = metrics['bond_CE'] + metrics['atom_conditional_CE']
        row = dict(epoch=epoch, train_loss=loss_sum / max(batches, 1), **metrics,
                   selection=selection, seconds=time.perf_counter() - start)
        history.append(row); p.log(f'{a.arm}: {row}')
        if selection < best:
            best = selection
            torch.save(dict(state=net.state_dict(), args=ck['args'], epoch=epoch,
                            continuation_args=vars(a), parent_epoch=ck['epoch']), target / 'best.pt')
        p.save_json(target / 'training.json', dict(args=vars(a), history=history,
                    selection='same criterion for both arms: allmask bond CE + true-B conditional atom CE; epoch 0 is eligible',
                    training_rule='original random mask CE + 0.5 complete-atom-mask CE; true B only in block arm',
                    scope='Continuation of developmental checkpoint; not independent test. No h/q labels imposed on corrupted B.'))


def test(a):
    j.tests()
    data = torch.load(j.HERE / 'features.pt', weights_only=False)['rows']
    net, _ = load_net(a.checkpoint, a.device)
    catalogue = j.make_catalogue(torch.load(j.HERE / 'sources.pt', weights_only=False)['train'])
    rows = data['val'][:4]; b = j.pack(rows, a.device)
    for weight in [0., 1.]:
        expected = j.sample(net, b, 1, 101, catalogue=catalogue if weight else None,
                            atoms_after_B=True, combine_catalogue=bool(weight))
        actual = draws(net, b, catalogue, 101, weight=weight)[0]
        assert all(torch.equal(x, y) for x, y in zip(expected, actual)), 'Baseline mismatch'
    baseline = draws(net, b, catalogue, 101)
    altered = dict(b, labels=torch.full_like(b['labels'], 3), atarget=torch.full_like(b['atarget'], 24))
    for name, settings in configs(['w1', 'adaptive', '4x2', '2x4', 'anchor']):
        x = draws(net, b, catalogue, 101, **settings)
        y = draws(net, altered, catalogue, 101, **settings)
        assert len(x) == 8
        assert all(torch.equal(xx, yy) for old, new in zip(x, y) for xx, yy in zip(old, new)), name
    p.log('improvement tests passed: baseline parity, raw budget=8, no inference label leakage, charge DP')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['test', 'evaluate', 'train'])
    parser.add_argument('--checkpoint', default=str(j.HERE / 'best.pt'))
    parser.add_argument('--device', default='cuda'); parser.add_argument('--seed', type=int, default=20261002)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--noises', nargs='+', type=float, default=[0., .1, .2])
    parser.add_argument('--configs', nargs='+', default=['w0', 'w025', 'w05', 'w1', 'w2', 'adaptive', '4x2', '2x4', 'anchor'])
    parser.add_argument('--output', default='scan.json')
    parser.add_argument('--arm', choices=['control_train', 'block_train'], default='block_train')
    parser.add_argument('--epochs', type=int, default=6); parser.add_argument('--lr', type=float, default=.0001)
    a = parser.parse_args(); torch.set_num_threads(4); OUT.mkdir(exist_ok=True)
    {'test': test, 'evaluate': evaluate, 'train': train}[a.action](a)


if __name__ == '__main__':
    main()
