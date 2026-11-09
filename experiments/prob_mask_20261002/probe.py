"""Isolated heavy-only masked graph pilot; never modifies released models.

Labels: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic;
5=MASK (an input token, never a predicted class).
"""
import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bondnet.model.bondnet import BondNet
from bondnet.data.dataset import _assign_split_label, _sanitize_with_mdl_aromaticity

HERE = Path(__file__).resolve().parent
MASK = 5
BT = {Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2,
      Chem.BondType.TRIPLE: 3, Chem.BondType.AROMATIC: 4}


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(p, data):
    Path(p).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_backbone(path, device):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    a = ck['args']
    model = BondNet(num_interactions=a['num_interactions'], hidden_size=a['hidden_size'],
                    cutoff=a['cutoff'], edge_embedding_size=a['edge_embedding_size'],
                    vector_norm_limit=a.get('vector_norm_limit', 0), dropout=0,
                    backbone=a.get('backbone', 'painn')).to(device)
    model.load_state_dict(ck['model_state_dict'], strict=True)
    return model.eval(), a


def geometry(row, sigma, rng, cutoff):
    xyz = row['coord'] + torch.tensor(rng.normal(0, sigma, row['coord'].shape), dtype=torch.float32)
    d = torch.cdist(xyz, xyz)
    pairs = torch.nonzero(torch.triu((d < cutoff) & (d > 0), diagonal=1))
    directed = torch.cat([pairs, pairs.flip(1)], dim=0)
    labels = torch.tensor([row['truth'].get(tuple(p), 0) for p in pairs.tolist()], dtype=torch.long)
    present = set(map(tuple, pairs.tolist()))
    missed = len(set(row['truth']) - present)
    return dict(row, coord=xyz, pairs=pairs, directed=directed, labels=labels,
                sigma=sigma, missed=missed)


@torch.no_grad()
def extract(model, rows, device):
    xyz, z, edges, offsets = [], [], [], []
    offset = 0
    for r in rows:
        offsets.append(offset)
        xyz.append(r['coord']); z.append(r['z']); edges.append(r['directed'] + offset)
        offset += len(r['z'])
    xyz = torch.cat(xyz).to(device); z = torch.cat(z).to(device)
    edges = torch.cat(edges).to(device)
    diff = xyz[edges[:, 1]] - xyz[edges[:, 0]]
    dist = diff.norm(dim=-1)
    _, _, feats = model.backbone(z, xyz, edges, diff, dist)
    conn = model.stage1(feats).reshape(-1)
    types = model.stage3(feats)[0]
    result, start = [], 0
    for r in rows:
        n = len(r['pairs']); stop = start + 2 * n
        f = feats[start:stop]; c = conn[start:stop]; t = types[start:stop]
        # Combine directed probabilities, preserving an independent baseline.
        pc = c.sigmoid()
        probs = torch.cat([(1-pc)[:, None], pc[:, None] * t.softmax(-1)], dim=-1)
        probs = (probs[:n] + probs[n:]) / 2
        hard = torch.where((pc[:n] > .5) | (pc[n:] > .5),
                           ((t[:n]+t[n:])/2).argmax(-1)+1, 0)
        result.append(dict(r, feat=((f[:n]+f[n:])/2).cpu().half(),
                           base=probs.clamp_min(1e-8).log().cpu(), hard=hard.cpu()))
        start = stop
    return result


def prepare(a):
    seed_all(a.seed)
    RDLogger.DisableLog('rdApp.*')
    rows = {'train': [], 'val': []}; seen = {'train': set(), 'val': set()}
    skips = {}; scanned = 0
    # Scan a fixed prefix, then seeded selection: no historical test samples.
    with open(a.source, 'rb') as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            if idx >= a.scan:
                break
            scanned += 1
            try:
                if mol is None:
                    raise ValueError('parse')
                group = int(mol.GetProp('geom_mol_idx'))
                split = _assign_split_label({'geom_mol_idx': group}, idx)
                if split == 'test':
                    continue
                mol = _sanitize_with_mdl_aromaticity(mol)
                mol = Chem.RemoveHs(mol)
                if any(x.GetAtomicNum() == 1 for x in mol.GetAtoms()):
                    raise ValueError('remaining_H')
                if any(x.GetFormalCharge() for x in mol.GetAtoms()):
                    raise ValueError('charged_out_of_scope')
                if len(Chem.GetMolFrags(mol)) != 1:
                    raise ValueError('fragment')
                smi = Chem.MolToSmiles(mol, isomericSmiles=False)
                if smi in seen[split]:
                    continue
                truth = {}
                for b in mol.GetBonds():
                    if b.GetBondType() not in BT:
                        raise ValueError('unsupported_bond')
                    truth[tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))] = BT[b.GetBondType()]
                seen[split].add(smi)
                rows[split].append({'id': group, 'source_index': idx, 'smiles': smi,
                                    'z': torch.tensor([x.GetAtomicNum() for x in mol.GetAtoms()]),
                                    'coord': torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32),
                                    'truth': truth})
            except Exception as exc:
                k = str(exc).split('\n')[0][:70]; skips[k] = skips.get(k, 0) + 1
    rng = random.Random(a.seed)
    for split in rows:
        rng.shuffle(rows[split])
    rows['val'] = rows['val'][:a.nval]
    val_identities = {r['smiles'] for r in rows['val']}
    overlap = sum(r['smiles'] in val_identities for r in rows['train'])
    rows['train'] = [r for r in rows['train'] if r['smiles'] not in val_identities][:a.ntrain]
    assert len(rows['train']) == a.ntrain and len(rows['val']) == a.nval, 'Increase --scan'
    assert not ({r['smiles'] for r in rows['train']} & val_identities)
    torch.save(rows, HERE / 'source_subset.pt')
    model, ca = load_backbone(a.checkpoint, a.device)
    outputs = {}; timings = {}
    for split in rows:
        perturbed = []
        for r in rows[split]:
            for sigma in a.noises:
                # The same coordinate noise is shared by every comparison.
                ng = np.random.default_rng(a.seed + 100000 * r['id'] + round(10000*sigma))
                perturbed.append(geometry(r, sigma, ng, ca['cutoff']))
        sync(); t0 = time.perf_counter(); out = []
        for i in range(0, len(perturbed), a.batch):
            out.extend(extract(model, perturbed[i:i+a.batch], a.device))
            if i % (a.batch * 40) == 0:
                log(f'feature cache {split}: {i}/{len(perturbed)}')
        sync(); timings[split] = time.perf_counter()-t0
        outputs[split] = out
    # An audit, not a claim that the pretrained network is exactly invariant.
    original = [geometry(r, 0, np.random.default_rng(0), ca['cutoff']) for r in rows['val'][:16]]
    q, _ = torch.linalg.qr(torch.randn(3, 3)); q[:, 0] *= torch.linalg.det(q)
    rotated = [dict(r, coord=r['coord'] @ q) for r in original]
    fo = extract(model, original, a.device); fr = extract(model, rotated, a.device)
    diffs = torch.cat([(x['base'].exp()-y['base'].exp()).abs().reshape(-1) for x, y in zip(fo, fr)])
    audit = {'molecules': len(fo), 'max_probability_change': diffs.max().item(),
             'mean_probability_change': diffs.mean().item(),
             'hard_graph_changes': sum(not torch.equal(x['hard'], y['hard']) for x, y in zip(fo, fr))}
    metadata = {'arguments': vars(a), 'checkpoint_sha256': hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                'scanned_records': scanned, 'skips': skips, 'train_identities_removed': overlap,
                'counts': {s: len(rows[s]) for s in rows}, 'feature_seconds': timings,
                'rotation_audit': audit,
                'scope': 'Neutral connected GEOM heavy-only validation pilot; q/h posterior not modelled.'}
    torch.save({'rows': outputs, 'metadata': metadata}, HERE / 'features.pt')
    save_json(HERE / 'prepare.json', metadata)
    log(f'prepared {metadata["counts"]}; rotation audit: {audit}')


class MaskedBondDecoder(nn.Module):
    """Small symmetric edge/incident-node message network over a partial graph."""
    def __init__(self, input_dim=360, width=96, layers=3, context=False):
        super().__init__()
        self.geo = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, width), nn.SiLU())
        self.state = nn.Embedding(6, width)
        self.context = context
        if context:
            self.chem = nn.Sequential(nn.Linear(14, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(3*width, width), nn.SiLU(),
                                                  nn.Linear(width, width)) for _ in range(layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.head = nn.Linear(width, 5)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, feat, base, pairs, states, nnode):
        h = self.geo(feat.float()) + self.state(states)
        i, j = pairs.unbind(1)
        degree = torch.zeros(nnode, 1, device=h.device)
        one = torch.ones(len(pairs), 1, device=h.device)
        degree.index_add_(0, i, one); degree.index_add_(0, j, one)
        if self.context:
            # Observed edge-state counts and lower-bound observed valence.
            # Aromatic contributes one here: do not impose a false 1.5 valence
            # cap on fused rings or hidden protonation/charge states.
            encoding = nn.functional.one_hot(states, 6).float()
            orders = torch.tensor([0., 1., 2., 3., 1., 0.], device=h.device)[states]
            encoding = torch.cat([encoding, orders[:, None]], -1)
            counts = torch.zeros(nnode, 7, device=h.device)
            counts.index_add_(0, i, encoding); counts.index_add_(0, j, encoding)
            h = h + self.chem(torch.cat([counts[i]+counts[j], (counts[i]-counts[j]).abs()], -1))
        for block, norm in zip(self.blocks, self.norms):
            nodes = torch.zeros(nnode, h.shape[1], device=h.device)
            nodes.index_add_(0, i, h); nodes.index_add_(0, j, h)
            nodes = nodes / degree.clamp_min(1)
            update = block(torch.cat([h, nodes[i]+nodes[j], (nodes[i]-nodes[j]).abs()], -1))
            h = norm(h + update)
        return base + self.head(h)


def pack(rows, device):
    pairs, offsets, offset = [], [], 0
    for r in rows:
        pairs.append(r['pairs']+offset); offsets.append(len(r['pairs']))
        offset += len(r['z'])
    return {'feat': torch.cat([r['feat'] for r in rows]).to(device).float(),
            'base': torch.cat([r['base'] for r in rows]).to(device),
            'pairs': torch.cat(pairs).to(device), 'nnode': offset,
            'labels': torch.cat([r['labels'] for r in rows]).to(device), 'lengths': offsets}


def call(net, b, states):
    return net(b['feat'], b['base'], b['pairs'], states, b['nnode'])


def masks(b):
    # Mix uniform masking, entropy-weighted masking, and atom-incident masking.
    entropy = -(b['base'].exp()*b['base']).sum(-1)/math.log(5)
    mask = torch.zeros_like(b['labels'], dtype=torch.bool)
    start = 0
    for n in b['lengths']:
        rate = .2 + .8*torch.rand((), device=mask.device)
        if torch.rand((), device=mask.device) < .2:
            rate = 1.
        p = torch.clamp(rate*(.5+entropy[start:start+n]), .05, 1.)
        mask[start:start+n] = torch.rand(n, device=mask.device) < p
        start += n
    if torch.rand((), device=mask.device) < .3:
        atommask = torch.rand(b['nnode'], device=mask.device) < .15
        mask |= atommask[b['pairs']].any(1)
    if not mask.any():
        mask[0] = True
    state = b['labels'].clone()
    corrupt = (~mask) & (torch.rand_like(entropy) < .05)
    state[corrupt] = torch.randint(0, 5, (int(corrupt.sum()),), device=state.device)
    state[mask] = MASK
    return state, mask | corrupt


@torch.no_grad()
def allmask_loss(net, rows, device, batch):
    total = count = 0
    for i in range(0, len(rows), batch):
        b = pack(rows[i:i+batch], device)
        out = call(net, b, torch.full_like(b['labels'], MASK))
        total += nn.functional.cross_entropy(out, b['labels'], reduction='sum').item()
        count += len(b['labels'])
    return total/count


def train(a):
    seed_all(a.seed); data = torch.load(HERE/'features.pt', weights_only=False)
    outdir = HERE/a.run; outdir.mkdir(exist_ok=True)
    rows = data['rows']; net = MaskedBondDecoder(width=a.width, layers=a.layers, context=a.context).to(a.device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []; best = float('inf')
    log(f'train decoder: {sum(p.numel() for p in net.parameters()):,} parameters')
    t0 = time.perf_counter()
    for epoch in range(1, a.epochs+1):
        net.train(); order = np.random.permutation(len(rows['train'])); total = count = 0
        for i in range(0, len(order), a.batch):
            b = pack([rows['train'][j] for j in order[i:i+a.batch]], a.device)
            state, supervised = masks(b)
            out = call(net, b, state)
            # Unweighted likelihood (weighted CE would distort sampling masses).
            loss = nn.functional.cross_entropy(out[supervised], b['labels'][supervised])
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            total += loss.item()*int(supervised.sum()); count += int(supervised.sum())
        net.eval(); val = allmask_loss(net, rows['val'], a.device, a.batch)
        item = {'epoch': epoch, 'masked_train_ce': total/count, 'allmask_val_ce': val,
                'elapsed_seconds': time.perf_counter()-t0}
        history.append(item); log(str(item))
        if val < best:
            best = val
            torch.save({'state': net.state_dict(), 'args': vars(a), 'epoch': epoch}, outdir/'decoder_best.pt')
        save_json(outdir/'training.json', {'history': history, 'selection': 'minimum all-MASK validation CE',
                                        'parameters': sum(p.numel() for p in net.parameters())})


@torch.no_grad()
def sample(net, b, rounds, stochastic=True, remask=False, generator=None):
    state = torch.full_like(b['labels'], MASK)
    if rounds == 1:
        probs = call(net, b, state).softmax(-1)
        return torch.multinomial(probs, 1, generator=generator).squeeze(1) if stochastic else probs.argmax(-1)
    lengths = torch.tensor(b['lengths'], device=state.device)
    starts = lengths.cumsum(0)-lengths
    gid = torch.repeat_interleave(torch.arange(len(lengths), device=state.device), lengths)
    local_index = torch.arange(len(state), device=state.device)-starts[gid]
    maxn = max(b['lengths'])
    for step in range(rounds):
        probs = call(net, b, state).softmax(-1)
        draw = torch.multinomial(probs, 1, generator=generator).squeeze(1) if stochastic else probs.argmax(-1)
        if step+1 == rounds:
            state = torch.where(state == MASK, draw, state)
            break
        remaining = (lengths.float()*math.cos(math.pi/2*(step+1)/rounds)).long()
        if remask:
            # Reconsider previously revealed edges as well. Visible corrupted
            # labels are supervised during training; no true labels are read.
            proposed = torch.where(state == MASK, draw, state)
            confidence = probs.gather(1, proposed[:, None]).squeeze(1)
            padded = torch.full((len(lengths), maxn), -torch.inf, device=state.device)
            padded[gid, local_index] = confidence
            order = padded.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(maxn, device=state.device)[None].expand_as(order))
            keep = rank[gid, local_index] < (lengths-remaining)[gid]
            state = torch.where(keep, proposed, MASK)
        else:
            confidence = probs.gather(1, draw[:, None]).squeeze(1)
            padded = torch.full((len(lengths), maxn), -torch.inf, device=state.device)
            padded[gid, local_index] = confidence.masked_fill(state != MASK, -torch.inf)
            order = padded.argsort(dim=1, descending=True)
            rank = torch.empty_like(order)
            rank.scatter_(1, order, torch.arange(maxn, device=state.device)[None].expand_as(order))
            pending = torch.zeros(len(lengths), device=state.device, dtype=torch.long)
            pending.index_add_(0, gid, (state == MASK).long())
            commit = (pending-remaining).clamp_min(0)
            choose = (state == MASK) & (rank[gid, local_index] < commit[gid])
            state = torch.where(choose, draw, state)
    assert not (state == MASK).any()
    return state


def molecule_smiles(row, labels):
    rw = Chem.RWMol()
    for z in row['z'].tolist():
        rw.AddAtom(Chem.Atom(z))
    types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for (i, j), c in zip(row['pairs'].tolist(), labels.tolist()):
        if c:
            rw.AddBond(i, j, types[c])
            if c == 4:
                rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        m = rw.GetMol(); Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, isomericSmiles=False), len(Chem.GetMolFrags(m)) == 1
    except Exception:
        return None, False


def evaluate(a):
    seed_all(a.seed); RDLogger.DisableLog('rdApp.*')
    data = torch.load(HERE/'features.pt', weights_only=False)
    outdir = HERE/a.run
    ck = torch.load(outdir/'decoder_best.pt', weights_only=False)
    net = MaskedBondDecoder(width=ck['args']['width'], layers=ck['args']['layers'], context=ck['args'].get('context', False)).to(a.device)
    net.load_state_dict(ck['state'], strict=True); net.eval()
    summary = []; records = []
    sources = torch.load(HERE/'source_subset.pt', weights_only=False)['val']
    truth_default_h = []
    for r in sources:
        tr = dict(r, pairs=torch.tensor(list(r['truth']), dtype=torch.long))
        smi, _ = molecule_smiles(tr, torch.tensor(list(r['truth'].values())))
        if smi != r['smiles']:
            truth_default_h.append({'id': r['id'], 'reference_has_nH': '[nH]' in r['smiles'],
                                    'default_h_sanitize_failed': smi is None})
    for sigma in sorted({r['sigma'] for r in data['rows']['val']}):
        if a.eval_noises is not None and sigma not in a.eval_noises:
            continue
        rows = [r for r in data['rows']['val'] if r['sigma'] == sigma]
        methods = [('frozen_hard', 0), ('decoder_argmax', 1), ('frozen_independent', 0), ('decoder_independent', 0)]+[('mask', n) for n in a.rounds]
        if a.remask:
            methods += [('remask', n) for n in a.rounds if n > 1]
        for method, rounds in methods:
            k = 1 if method in ['frozen_hard', 'decoder_argmax'] else a.candidates
            generated = []; infer_seconds = 0
            for i in range(0, len(rows), a.batch):
                batchrows = rows[i:i+a.batch]; b = pack(batchrows, a.device)
                generators = [torch.Generator(device=a.device).manual_seed(a.seed+round(sigma*10000)+i*97+j*1000000) for j in range(k)]
                sync(); start = time.perf_counter()
                if method == 'frozen_hard':
                    draws = [torch.cat([r['hard'] for r in batchrows]).to(a.device)]
                elif method == 'decoder_argmax':
                    draws = [call(net, b, torch.full_like(b['labels'], MASK)).argmax(-1)]
                elif method.endswith('independent'):
                    logits = b['base'] if method == 'frozen_independent' else call(net, b, torch.full_like(b['labels'], MASK))
                    probs = logits.softmax(-1)
                    draws = [torch.multinomial(probs, 1, generator=g).squeeze(1) for g in generators]
                else:
                    draws = [sample(net, b, rounds, remask=method == 'remask', generator=g) for g in generators]
                sync(); infer_seconds += time.perf_counter()-start
                splitdraws = [d.cpu().split(b['lengths']) for d in draws]
                for j, r in enumerate(batchrows):
                    generated.append([d[j] for d in splitdraws])
            strict = species = valid = connected = unique = top1 = 0
            if method == 'decoder_independent':
                independent_draws = generated
            elif method == 'mask' and rounds == 1:
                assert all(torch.equal(x, y) for old, new in zip(independent_draws, generated)
                           for x, y in zip(old, new)), 'One-round mask must match independent sampling under common seeds'
            for r, draws in zip(rows, generated):
                exacts = [bool(torch.equal(d, r['labels']) and r['missed'] == 0) for d in draws]
                sm = [molecule_smiles(r, d) for d in draws]
                strict += any(exacts); top1 += exacts[0]
                species += any(s == r['smiles'] for s, _ in sm)
                valid += sum(s is not None for s, _ in sm); connected += sum(c for _, c in sm)
                unique += len({s for s, _ in sm if s is not None})
                records.append({'id': r['id'], 'sigma': sigma, 'method': method, 'rounds': rounds,
                                'strict_hit': any(exacts), 'species_hit': any(s == r['smiles'] for s, _ in sm),
                                'first_exact': exacts[0], 'valid_unique': len({s for s, _ in sm if s is not None})})
            n = len(rows)
            item = {'sigma': sigma, 'method': method, 'rounds': rounds, 'n': n, 'budget': k,
                    'strict_coverage': strict/n, 'species_coverage': species/n, 'first_draw_exact': top1/n,
                    'sanitize_fraction': valid/(n*k), 'connected_valid_fraction': connected/(n*k),
                    'mean_valid_unique': unique/n, 'decoder_ms_per_molecule_set': 1000*infer_seconds/n,
                    'candidate_graph_complete_fraction': sum(r['missed'] == 0 for r in rows)/n}
            summary.append(item); log(str(item))
    save_json(outdir/a.evaluation_name, {'checkpoint_epoch': ck['epoch'], 'arguments': vars(a),
                                     'summary': summary, 'records': records,
                                     'truth_graph_default_H_failures': truth_default_h,
                                     'species_note': 'Default-H reconstruction diagnostic, NOT joint species posterior recovery; even the true heavy graph can fail.',
                                     'sampling_note': 'Common per-candidate generators across methods; one-round mask and decoder-independent samples checked bit-identical.',
                                     'timing_note': 'Decoder-only CUDA-synchronized latency; geometry cached; excludes RDKit, transfers and startup.'})


def tests():
    seed_all(9)
    net = MaskedBondDecoder(input_dim=8, width=16, layers=2, context=True)
    nn.init.normal_(net.head.weight, std=.1)
    feat = torch.randn(3, 8); base = torch.randn(3, 5)
    pairs = torch.tensor([[0, 1], [1, 2], [2, 0]])
    states = torch.tensor([MASK, 1, 0])
    a = net(feat, base, pairs, states, 3)
    b = net(feat, base, pairs.flip(1), states, 3)
    assert torch.allclose(a, b, atol=1e-6), 'Undirected symmetry failed'
    changed = states.clone(); changed[1] = 2
    c = net(feat, base, pairs, changed, 3)
    assert not torch.allclose(a[0], c[0]), 'No incident-edge dependency'
    batch = {'feat': feat, 'base': base, 'pairs': pairs, 'nnode': 3, 'labels': torch.zeros(3, dtype=torch.long), 'lengths': [3]}
    for n in [1, 2, 4, 8]:
        for remask in [False, True]:
            seed_all(42)
            out = sample(net, batch, n, remask=remask)
            assert ((out >= 0) & (out < MASK)).all()
            # Labels are carried for scoring only; generation must not read values.
            altered = dict(batch, labels=torch.full_like(batch['labels'], 3))
            seed_all(42)
            assert torch.equal(out, sample(net, altered, n, remask=remask)), 'Reference-label leakage'
    log('tests passed: pair symmetry, graph dependency, distinct MASK, all-round completion, no label leakage')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'train', 'evaluate', 'test'])
    p.add_argument('--source', default=str(ROOT/'data/geom_drugs_all_random1_rel10.sdf'))
    p.add_argument('--checkpoint', default=str(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    p.add_argument('--seed', type=int, default=20261002)
    p.add_argument('--device', default='cuda')
    p.add_argument('--ntrain', type=int, default=3000); p.add_argument('--nval', type=int, default=300)
    p.add_argument('--scan', type=int, default=12000)
    p.add_argument('--noises', type=float, nargs='+', default=[0., .1, .2])
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=12); p.add_argument('--width', type=int, default=96)
    p.add_argument('--layers', type=int, default=3); p.add_argument('--lr', type=float, default=.0003)
    p.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--candidates', type=int, default=8)
    p.add_argument('--run', default='.')
    p.add_argument('--context', action='store_true')
    p.add_argument('--remask', action='store_true')
    p.add_argument('--eval-noises', type=float, nargs='+', default=None)
    p.add_argument('--evaluation-name', default='evaluation.json')
    a = p.parse_args()
    torch.set_num_threads(4)
    {'prepare': prepare, 'train': train, 'evaluate': evaluate, 'test': lambda _: tests()}[a.action](a)


if __name__ == '__main__':
    main()
