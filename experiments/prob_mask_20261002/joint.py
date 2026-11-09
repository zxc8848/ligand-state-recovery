"""Joint heavy-edge B and atomic (h,q) masked completion pilot.

No source/reference atom state is an inference feature. Total Q is an explicitly
provided condition. Formal q=-2..2 and total attached h=0..4 form 25 joint states.
Exact batched dynamic programming enforces sum(q)=Q for every sampling method.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger

import probe as p

HERE = p.HERE/'joint_v1'
AMASK = 25
BMASK = 5


def skeleton(mol):
    """Connectivity/element identity, blinded to h/q/bond type/stereochemistry."""
    rw = Chem.RWMol()
    for a in mol.GetAtoms():
        b = Chem.Atom(a.GetAtomicNum()); b.SetNoImplicit(True); rw.AddAtom(b)
    for b in mol.GetBonds():
        rw.AddBond(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), Chem.BondType.SINGLE)
    return Chem.MolToSmiles(rw.GetMol(), isomericSmiles=False)


def make_row(mol, key, index, domain):
    mol = p._sanitize_with_mdl_aromaticity(mol)
    # GetTotalNumHs(includeNeighbors=True) counts BOTH explicit-neighbor and
    # implicit/property hydrogens, unlike an explicit-neighbor-only target.
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            atom.SetIntProp('_joint_h', atom.GetTotalNumHs(includeNeighbors=True))
    mol = Chem.RemoveHs(mol)
    if any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()):
        raise ValueError('remaining_H')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('radical_reference')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('fragment_reference')
    z = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    h = torch.tensor([a.GetIntProp('_joint_h') for a in mol.GetAtoms()])
    q = torch.tensor([a.GetFormalCharge() for a in mol.GetAtoms()])
    if (h > 4).any() or (q.abs() > 2).any() or (z > 118).any():
        raise ValueError('out_of_vocabulary')
    truth = {}
    for b in mol.GetBonds():
        if b.GetBondType() not in p.BT:
            raise ValueError('unsupported_bond')
        truth[tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))] = p.BT[b.GetBondType()]
    row = {'id': key, 'source_index': index, 'domain': domain, 'skeleton': skeleton(mol),
           'smiles': Chem.MolToSmiles(mol, isomericSmiles=False), 'z': z,
           'h': h, 'q': q, 'atom_target': (q+2)*5+h, 'Q': int(q.sum()),
           'coord': torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32),
           'truth': truth}
    full = dict(row, pairs=torch.tensor(list(truth), dtype=torch.long))
    smi, valid = build_species(full, torch.tensor(list(truth.values())), row['atom_target'])
    if not valid or smi != row['smiles']:
        raise ValueError('truth_Bhq_roundtrip_failed')
    return row


def build_species(row, bond, atom_state):
    rw = Chem.RWMol()
    for z, state in zip(row['z'].tolist(), atom_state.tolist()):
        atom = Chem.Atom(z); atom.SetFormalCharge(state//5-2)
        atom.SetNumExplicitHs(state % 5); atom.SetNoImplicit(True)
        rw.AddAtom(atom)
    types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for (i, j), c in zip(row['pairs'].tolist(), bond.tolist()):
        if c:
            rw.AddBond(i, j, types[c])
            if c == 4:
                rw.GetAtomWithIdx(i).SetIsAromatic(True); rw.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        mol = rw.GetMol(); Chem.SanitizeMol(mol)
        closed = not any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
        connected = len(Chem.GetMolFrags(mol)) == 1
        return Chem.MolToSmiles(mol, isomericSmiles=False), closed and connected
    except Exception:
        return None, False


@torch.no_grad()
def extract(model, rows, device):
    z, xyz, edges, offset = [], [], [], 0
    for r in rows:
        z.append(r['z']); xyz.append(r['coord']); edges.append(r['directed']+offset)
        offset += len(r['z'])
    z = torch.cat(z).to(device); xyz = torch.cat(xyz).to(device); edges = torch.cat(edges).to(device)
    diff = xyz[edges[:, 1]]-xyz[edges[:, 0]]
    node, _, ef = model.backbone(z, xyz, edges, diff, diff.norm(dim=-1))
    conn = model.stage1(ef).reshape(-1); types = model.stage3(ef)[0]
    out, es, ns = [], 0, 0
    for r in rows:
        ne = len(r['pairs']); na = len(r['z'])
        feats = ef[es:es+2*ne]; c = conn[es:es+2*ne]; t = types[es:es+2*ne]
        prob = torch.cat([(1-c.sigmoid())[:, None], c.sigmoid()[:, None]*t.softmax(-1)], -1)
        prob = ((prob[:ne]+prob[ne:])/2).clamp_min(1e-8)
        out.append(dict(r, feat=((feats[:ne]+feats[ne:])/2).cpu().half(),
                        node_feat=node[ns:ns+na].cpu().half(), base=prob.log().cpu()))
        es += 2*ne; ns += na
    return out


def prepare(a):
    HERE.mkdir(exist_ok=True); p.seed_all(a.seed); RDLogger.DisableLog('rdApp.*')
    previous = torch.load(p.HERE/'source_subset.pt', weights_only=False)
    wanted = {r['source_index']: s for s in previous for r in previous[s]}
    rows = {'train': [], 'val': []}; skipped = collections.Counter()
    with open(a.geom, 'rb') as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            if idx > max(wanted):
                break
            if idx not in wanted:
                continue
            try:
                row = make_row(mol, int(mol.GetProp('geom_mol_idx')), idx, 'geom_core')
                rows[wanted[idx]].append(row)
            except Exception as exc:
                skipped[str(exc).split('\n')[0][:80]] += 1
    # Fixed connectivity-only group hash: protonation/charge/tautomer variants
    # cannot cross train and validation. Extra-source splits are freshly made;
    # this is a development corpus, not an untouched historical test claim.
    pools = {(s, bucket): [] for s in ['train', 'val'] for bucket in ['net_charged', 'internal_charge']}
    seen = set(); scanned = 0
    p.log(f'GEOM reference reconstruction completed: {[len(x) for x in rows.values()]}')
    with open(a.diverse, 'rb') as handle:
        supplier = Chem.ForwardSDMolSupplier(handle, sanitize=False, removeHs=False)
        for idx, mol in enumerate(supplier):
            scanned += 1
            if idx % 5000 == 0:
                p.log(f'diverse charge scan {idx}; pool sizes {[len(x) for x in pools.values()]}')
            try:
                if mol is None:
                    raise ValueError('diverse_parse')
                mol = p._sanitize_with_mdl_aromaticity(mol)
                if not any(x.GetFormalCharge() for x in mol.GetAtoms()):
                    continue
                row = make_row(mol, 1000000000+idx, idx, 'diverse_charge')
                if row['smiles'] in seen:
                    continue
                seen.add(row['smiles'])
                split = p._assign_split_label({'geom_mol_idx': row['skeleton']}, idx)
                if split == 'test':
                    continue
                bucket = 'net_charged' if row['Q'] else 'internal_charge'
                pools[(split, bucket)].append(row)
            except Exception as exc:
                skipped[str(exc).split('\n')[0][:80]] += 1
    rng = random.Random(a.seed)
    core_val_skeletons = {r['skeleton'] for r in rows['val']}
    for pool in pools.values():
        rng.shuffle(pool)
    # Challenge set intentionally enriched for net-charged cases; stratified
    # results, not a natural-population validity estimate.
    for bucket, nv in [('net_charged', a.charge_val), ('internal_charge', a.internal_val)]:
        chosen = [r for r in pools[('val', bucket)] if r['skeleton'] not in core_val_skeletons][:nv]
        rows['val'].extend(chosen)
    val_skeletons = {r['skeleton'] for r in rows['val']}
    removed = sum(r['skeleton'] in val_skeletons for r in rows['train'])
    rows['train'] = [r for r in rows['train'] if r['skeleton'] not in val_skeletons]
    for bucket, nt in [('net_charged', a.charge_train), ('internal_charge', a.internal_train)]:
        chosen = [r for r in pools[('train', bucket)] if r['skeleton'] not in val_skeletons][:nt]
        rows['train'].extend(chosen)
    assert not ({r['skeleton'] for r in rows['train']} & val_skeletons)
    assert sum(r['Q'] != 0 for r in rows['train']) > 100, 'Insufficient charge coverage'
    stats = {}
    for split, rr in rows.items():
        stats[split] = {'n': len(rr), 'domains': dict(collections.Counter(r['domain'] for r in rr)),
                        'total_charge_hist': dict(collections.Counter(r['Q'] for r in rr)),
                        'atomic_charge_hist': dict(collections.Counter(int(q) for r in rr for q in r['q'])),
                        'H_hist': dict(collections.Counter(int(h) for r in rr for h in r['h'])),
                        'nH_species': sum('[nH]' in r['smiles'] for r in rr)}
    torch.save(rows, HERE/'sources.pt')
    model, ca = p.load_backbone(a.checkpoint, a.device)
    cached = {}; timings = {}
    for split in rows:
        geos = []
        for r in rows[split]:
            for sigma in a.noises:
                ng = np.random.default_rng(a.seed+100000*r['id']+round(10000*sigma))
                geos.append(p.geometry(r, sigma, ng, ca['cutoff']))
        t0 = time.perf_counter(); out = []
        for i in range(0, len(geos), a.batch):
            out.extend(extract(model, geos[i:i+a.batch], a.device))
            if i % (a.batch*40) == 0:
                p.log(f'joint cache {split}: {i}/{len(geos)}')
        p.sync(); timings[split] = time.perf_counter()-t0; cached[split] = out
    meta = {'arguments': vars(a), 'stats': stats, 'skips': dict(skipped), 'diverse_scanned': scanned,
            'train_core_skeletons_removed': removed, 'feature_seconds': timings,
            'checkpoint_sha256': hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
            'leakage_rule': 'Connectivity+element-only skeleton disjoint; blinded to q/h/bond type/stereo.',
            'known_input': 'Z,R,Q only; atomic h/q are target labels, not input features.',
            'roundtrip': 'All retained reference B,h,q reconstruct identical non-isomeric species.'}
    torch.save({'rows': cached, 'metadata': meta}, HERE/'features.pt'); p.save_json(HERE/'prepare.json', meta)
    p.log(f'joint prepared: {stats}')


class JointNet(nn.Module):
    def __init__(self, width=96, layers=2):
        super().__init__()
        self.edge = p.MaskedBondDecoder(width=96, layers=3, context=True)
        self.eg = nn.Sequential(nn.LayerNorm(360), nn.Linear(360, width), nn.SiLU())
        self.ng = nn.Sequential(nn.LayerNorm(360), nn.Linear(360, width), nn.SiLU())
        self.zemb = nn.Embedding(119, width)
        self.aemb = nn.Embedding(26, width); self.bemb = nn.Embedding(6, width)
        self.qembed = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.eblocks = nn.ModuleList([nn.Sequential(nn.Linear(3*width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(layers)])
        self.nblocks = nn.ModuleList([nn.Sequential(nn.Linear(2*width+6, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(layers)])
        self.enorm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.nnorm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.bhead = nn.Linear(width, 5); self.ahead = nn.Linear(width, 25)
        nn.init.zeros_(self.bhead.weight); nn.init.zeros_(self.bhead.bias)

    def forward(self, b, bs, ats):
        i, j = b['pairs'].unbind(1)
        h = self.eg(b['feat'])+self.bemb(bs)
        Q = b['Q'][b['atom_graph']].float(); count = b['atom_counts'][b['atom_graph']].float()
        qfeature = torch.stack([Q, Q.abs(), Q/count], -1)
        n = self.ng(b['node_feat'])+self.zemb(b['z'])+self.aemb(ats)+self.qembed(qfeature)
        deg = torch.zeros(len(n), 1, device=n.device); one = torch.ones(len(h), 1, device=n.device)
        deg.index_add_(0, i, one); deg.index_add_(0, j, one)
        code = nn.functional.one_hot(bs, 6).float()
        counts = torch.zeros(len(n), 6, device=n.device)
        counts.index_add_(0, i, code); counts.index_add_(0, j, code)
        for eb, nb, en, nnorm in zip(self.eblocks, self.nblocks, self.enorm, self.nnorm):
            h = en(h+eb(torch.cat([h, n[i]+n[j], (n[i]-n[j]).abs()], -1)))
            agg = torch.zeros_like(n); agg.index_add_(0, i, h); agg.index_add_(0, j, h)
            n = nnorm(n+nb(torch.cat([n, agg/deg.clamp_min(1), counts], -1)))
        blogits = self.edge(b['feat'], b['base'], b['pairs'], bs, len(n))+self.bhead(h)
        return blogits, self.ahead(n)


def pack(rows, device):
    b = p.pack(rows, device); ncounts = [len(r['z']) for r in rows]
    b.update({'node_feat': torch.cat([r['node_feat'] for r in rows]).to(device).float(),
              'z': torch.cat([r['z'] for r in rows]).to(device),
              'atarget': torch.cat([r['atom_target'] for r in rows]).to(device),
              'Q': torch.tensor([r['Q'] for r in rows], device=device),
              'atom_counts': torch.tensor(ncounts, device=device),
              'atom_graph': torch.repeat_interleave(torch.arange(len(rows), device=device), torch.tensor(ncounts, device=device)),
              'atom_lengths': ncounts})
    return b


def corrupt_states(b):
    bs, bsuper = p.masks(b)
    ats = b['atarget'].clone(); amask = torch.zeros_like(ats, dtype=torch.bool)
    start = 0
    for n in b['atom_lengths']:
        rate = .2+.8*torch.rand((), device=ats.device)
        if torch.rand((), device=ats.device) < .3:
            rate = 1.
        amask[start:start+n] = torch.rand(n, device=ats.device) < rate; start += n
    # Correlated atom/star masking exposes uncertain local chemistry jointly.
    if torch.rand((), device=ats.device) < .4:
        stars = amask[b['pairs']].any(1)
        bs[stars] = BMASK; bsuper |= stars
    corrupt = (~amask) & (torch.rand(len(ats), device=ats.device) < .03)
    ats[corrupt] = torch.randint(0, 25, (int(corrupt.sum()),), device=ats.device)
    ats[amask] = AMASK
    if not amask.any():
        ats[0] = AMASK; amask[0] = True
    # Explicit whole-state all-MASK examples match the start of inference.
    if torch.rand((), device=ats.device) < .2:
        ats.fill_(AMASK); bs.fill_(BMASK); amask.fill_(True); bsuper.fill_(True)
    return bs, ats, bsuper, amask | corrupt


@torch.no_grad()
def val_ce(net, rows, device, batch):
    bt = at = bn = an = 0
    for i in range(0, len(rows), batch):
        b = pack(rows[i:i+batch], device)
        bl, al = net(b, torch.full_like(b['labels'], BMASK), torch.full_like(b['atarget'], AMASK))
        bt += nn.functional.cross_entropy(bl, b['labels'], reduction='sum').item(); bn += len(bl)
        at += nn.functional.cross_entropy(al, b['atarget'], reduction='sum').item(); an += len(al)
    return bt/bn, at/an


def train(a):
    p.seed_all(a.seed); data = torch.load(HERE/'features.pt', weights_only=False)['rows']
    net = JointNet(width=a.width, layers=a.layers).to(a.device)
    ck = torch.load(p.HERE/'pilot_v2/decoder_best.pt', weights_only=False)
    net.edge.load_state_dict(ck['state'], strict=True)
    optimizer = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    history = []; best = float('inf'); t0 = time.perf_counter()
    p.log(f'joint train parameters: {sum(x.numel() for x in net.parameters()):,}')
    for epoch in range(1, a.epochs+1):
        net.train(); order = np.random.permutation(len(data['train'])); loss_total = batches = 0
        for i in range(0, len(order), a.batch):
            b = pack([data['train'][j] for j in order[i:i+a.batch]], a.device)
            bs, ats, bsuper, asuper = corrupt_states(b)
            bl, al = net(b, bs, ats)
            loss = nn.functional.cross_entropy(bl[bsuper], b['labels'][bsuper])+nn.functional.cross_entropy(al[asuper], b['atarget'][asuper])
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 2.); optimizer.step()
            loss_total += loss.item(); batches += 1
        net.eval(); vb, va = val_ce(net, data['val'], a.device, a.batch)
        row = {'epoch': epoch, 'train_loss': loss_total/batches, 'val_bond_CE': vb,
               'val_atom_joint_CE': va, 'seconds': time.perf_counter()-t0}
        history.append(row); p.log(str(row))
        if vb+va < best:
            best = vb+va
            torch.save({'state': net.state_dict(), 'args': vars(a), 'epoch': epoch}, HERE/'best.pt')
        p.save_json(HERE/'training.json', {'history': history, 'selection': 'all-mask bond CE + joint-atom CE',
                                         'parameters': sum(x.numel() for x in net.parameters())})


def constrained_atoms(logits, state, lengths, total_Q, rng, greedy=False):
    """Exact conditional draws for product atomic probabilities subject to Q.

    Batched suffix DP, float64 with per-row rescaling; visible states held fixed.
    No element/h/q target labels, bond truth or reference SMILES are consulted.
    """
    values = logits.detach().double().cpu().numpy(); current = state.cpu().numpy()
    Q = np.asarray(total_Q.cpu(), dtype=int); batch = len(lengths); nmax = max(lengths)
    center = 2*nmax; width = 4*nmax+1
    if np.any(np.abs(Q) > 2*np.array(lengths)):
        raise ValueError('Q infeasible for q vocabulary')
    probs = np.zeros((batch, nmax, 25), dtype=np.float64); probs[:, :, 10] = 1.
    start = 0
    for g, n in enumerate(lengths):
        x = values[start:start+n]; x = np.exp(x-x.max(1, keepdims=True)); x /= x.sum(1, keepdims=True)
        observed = current[start:start+n] != AMASK
        x[observed] = np.eye(25)[current[start:start+n][observed]]
        probs[g, :n] = x; start += n
    joint = probs.reshape(batch, nmax, 5, 5); qprob = joint.sum(-1)
    suffix = np.zeros((nmax+1, batch, width), dtype=np.float64); suffix[-1, :, center] = 1.
    for k in range(nmax-1, -1, -1):
        nxt = suffix[k+1]; out = suffix[k]
        for qi, q in enumerate(range(-2, 3)):
            if q < 0:
                out[:, :q] += qprob[:, k, qi, None]*nxt[:, -q:]
            elif q > 0:
                out[:, q:] += qprob[:, k, qi, None]*nxt[:, :-q]
            else:
                out += qprob[:, k, qi, None]*nxt
        out /= np.maximum(out.max(1, keepdims=True), 1e-300)
    if np.any(suffix[0, np.arange(batch), center+Q] == 0):
        raise ValueError('Visible atom states make Q infeasible')
    remain = Q.copy(); draws = np.zeros((batch, nmax), dtype=np.int64)
    for k in range(nmax):
        index = center+remain[:, None]-np.arange(-2, 3)[None]
        feasible = (index >= 0) & (index < width)
        weights = qprob[:, k]*suffix[k+1, np.arange(batch)[:, None], index.clip(0, width-1)]*feasible
        weights /= weights.sum(1, keepdims=True)
        qi = weights.argmax(1) if greedy else (rng.random(batch)[:, None] > weights.cumsum(1)).sum(1).clip(0, 4)
        hp = joint[np.arange(batch), k, qi]; hp = hp/hp.sum(1, keepdims=True)
        h = hp.argmax(1) if greedy else (rng.random(batch)[:, None] > hp.cumsum(1)).sum(1).clip(0, 4)
        draws[:, k] = qi*5+h; remain -= qi-2
    assert np.all(remain == 0)
    output = np.concatenate([draws[g, :n] for g, n in enumerate(lengths)])
    return torch.tensor(output, dtype=torch.long, device=logits.device)


def reveal(state, draw, confidence, lengths, remaining_fraction, token):
    sizes = torch.tensor(lengths, device=state.device); starts = sizes.cumsum(0)-sizes
    gid = torch.repeat_interleave(torch.arange(len(sizes), device=state.device), sizes)
    local = torch.arange(len(state), device=state.device)-starts[gid]; nmax = max(lengths)
    score = torch.full((len(sizes), nmax), -torch.inf, device=state.device)
    score[gid, local] = confidence.masked_fill(state != token, -torch.inf)
    order = score.argsort(1, descending=True); ranks = torch.empty_like(order)
    ranks.scatter_(1, order, torch.arange(nmax, device=state.device)[None].expand_as(order))
    pending = torch.zeros(len(sizes), device=state.device, dtype=torch.long)
    pending.index_add_(0, gid, (state == token).long())
    remaining = (sizes.float()*remaining_fraction).long(); commits = (pending-remaining).clamp_min(0)
    chosen = (state == token) & (ranks[gid, local] < commits[gid])
    return torch.where(chosen, draw, state)


def make_catalogue(rows):
    exact = {}; coarse = {}; by_z = {}
    for r in rows:
        neighbors = [[] for _ in r['z']]
        z = r['z'].tolist()
        for (i, j), c in r['truth'].items():
            neighbors[i].append((z[j], c)); neighbors[j].append((z[i], c))
        for i, (zi, state) in enumerate(zip(z, r['atom_target'].tolist())):
            key = (zi, tuple(sorted(neighbors[i])))
            ck = (zi, tuple(sorted(c for _, c in neighbors[i])))
            for table, k in [(exact, key), (coarse, ck), (by_z, zi)]:
                if k not in table:
                    table[k] = np.zeros(25, dtype=float)
                table[k][state] += 1
    return exact, coarse, by_z


def catalogue_logits(b, bd, catalogue):
    """Training-only local graph frequencies; no node target values consulted.

    A small pseudocount preserves support in unseen/Q-infeasible local states.
    This is a soft prior, not a hard catalogue/MILP equivalence baseline.
    """
    exact, coarse, by_z = catalogue
    z = b['z'].cpu().tolist(); pairs = b['pairs'].cpu().tolist(); labels = bd.cpu().tolist()
    neighbors = [[] for _ in z]
    for (i, j), c in zip(pairs, labels):
        if c:
            neighbors[i].append((z[j], c)); neighbors[j].append((z[i], c))
    logits = []
    for zi, nei in zip(z, neighbors):
        counts = exact.get((zi, tuple(sorted(nei))))
        if counts is None:
            counts = coarse.get((zi, tuple(sorted(c for _, c in nei))))
        if counts is None:
            counts = by_z.get(zi, np.ones(25))
        probs = counts+1e-4; probs /= probs.sum()
        logits.append(np.log(probs))
    return torch.tensor(np.array(logits), device=b['feat'].device, dtype=torch.float32)


@torch.no_grad()
def sample(net, b, rounds, seed, greedy=False, cached_logits=None, catalogue=None,
           atoms_after_B=False, combine_catalogue=False):
    bs = torch.full_like(b['labels'], BMASK); ats = torch.full_like(b['atarget'], AMASK)
    gen = torch.Generator(device=b['feat'].device).manual_seed(seed); rng = np.random.default_rng(seed)
    for step in range(rounds):
        bl, al = cached_logits if cached_logits is not None else net(b, bs, ats)
        bp = bl.softmax(-1)
        bd = bp.argmax(-1) if greedy else torch.multinomial(bp, 1, generator=gen).squeeze(1)
        if atoms_after_B:
            assert rounds == 1
            # All atoms remain masked until the complete *sampled* B is seen.
            # This changes the factorization, not the coordinate distribution.
            _, al = net(b, bd, ats)
        if catalogue is not None:
            assert rounds == 1
            prior = catalogue_logits(b, bd, catalogue)
            al = al+prior if combine_catalogue else prior
        ad = constrained_atoms(al, ats, b['atom_lengths'], b['Q'], rng, greedy)
        if step+1 == rounds:
            bs = torch.where(bs == BMASK, bd, bs); ats = torch.where(ats == AMASK, ad, ats)
            break
        fraction = math.cos(math.pi/2*(step+1)/rounds)
        bs = reveal(bs, bd, bp.gather(1, bd[:, None]).squeeze(1), b['lengths'], fraction, BMASK)
        ap = al.softmax(-1)
        ats = reveal(ats, ad, ap.gather(1, ad[:, None]).squeeze(1), b['atom_lengths'], fraction, AMASK)
    assert (bs != BMASK).all() and (ats != AMASK).all()
    assert torch.equal(torch.zeros_like(b['Q']).index_add_(0, b['atom_graph'], ats//5-2), b['Q'])
    return bs, ats


def eval_run(a):
    RDLogger.DisableLog('rdApp.*'); p.seed_all(a.seed)
    data = torch.load(HERE/'features.pt', weights_only=False)['rows']
    ck = torch.load(HERE/'best.pt', weights_only=False)
    net = JointNet(ck['args']['width'], ck['args']['layers']).to(a.device); net.load_state_dict(ck['state']); net.eval()
    catalogue = make_catalogue(torch.load(HERE/'sources.pt', weights_only=False)['train'])
    summary = []; allrecords = []
    for sigma in a.eval_noises:
        rows = [r for r in data['val'] if r['sigma'] == sigma]
        methods = [('greedy_joint', 1), ('independent_joint', 1), ('B_catalogue', 1),
                   ('B_then_HQ', 1), ('B_then_HQ_catalogue', 1)]+[('mask_joint', n) for n in a.rounds]
        if a.methods is not None:
            methods = [(m, n) for m, n in methods if m in a.methods]
        for method, rounds in methods:
            k = 1 if method == 'greedy_joint' else a.candidates
            results = []; seconds = 0
            for i in range(0, len(rows), a.batch):
                rr = rows[i:i+a.batch]; b = pack(rr, a.device)
                cached = None
                p.sync(); t0 = time.perf_counter()
                if rounds == 1:
                    cached = net(b, torch.full_like(b['labels'], BMASK), torch.full_like(b['atarget'], AMASK))
                draws = [sample(net, b, rounds, a.seed+round(sigma*10000)+i*97+j*1000000,
                                greedy=method == 'greedy_joint', cached_logits=cached,
                                catalogue=catalogue if method in ['B_catalogue', 'B_then_HQ_catalogue'] else None,
                                atoms_after_B=method in ['B_then_HQ', 'B_then_HQ_catalogue'],
                                combine_catalogue=method == 'B_then_HQ_catalogue') for j in range(k)]
                p.sync(); seconds += time.perf_counter()-t0
                bd = [x.cpu().split(b['lengths']) for x, _ in draws]
                ad = [x.cpu().split(b['atom_lengths']) for _, x in draws]
                for g, r in enumerate(rr):
                    results.append([(x[g], y[g]) for x, y in zip(bd, ad)])
            if method == 'independent_joint':
                independent_results = results
            elif method == 'mask_joint' and rounds == 1:
                assert all(torch.equal(b1, b2) and torch.equal(a1, a2)
                           for old, new in zip(independent_results, results) for (b1, a1), (b2, a2) in zip(old, new))
            records = []
            for r, draws in zip(rows, results):
                hits = []; graphs = []; joint = []; valid = []; smiles = []
                for bd, ad in draws:
                    smi, good = build_species(r, bd, ad)
                    hits.append(smi == r['smiles']); valid.append(good); smiles.append(smi if good else None)
                    graph = torch.equal(bd, r['labels']) and r['missed'] == 0
                    graphs.append(graph); joint.append(graph and torch.equal(ad, r['atom_target']))
                records.append({'id': r['id'], 'domain': r['domain'], 'Q': r['Q'], 'has_nH': '[nH]' in r['smiles'],
                                'skeleton': r['skeleton'],
                                'method': method, 'rounds': rounds, 'sigma': sigma,
                                'species_hits': hits, 'graph_hits': graphs, 'joint_hits': joint,
                                'valid_count': sum(valid), 'valid_unique': len({s for s in smiles if s is not None}),
                                'reference_smiles': r['smiles'], 'candidate_species': smiles,
                                'first_h_exact': torch.equal(draws[0][1] % 5, r['h']),
                                'first_q_exact': torch.equal(draws[0][1]//5-2, r['q']),
                                'first_h_correct_atoms': int((draws[0][1] % 5 == r['h']).sum()),
                                'first_q_correct_atoms': int((draws[0][1]//5-2 == r['q']).sum()),
                                'q_confusion': torch.bincount((r['q']+2)*5+draws[0][1]//5, minlength=25).reshape(5, 5).tolist(),
                                'atom_n': len(r['z']),
                                'Q_preserved': all(int((ad//5-2).sum()) == r['Q'] for _, ad in draws)})
            allrecords.extend(records)
            subsets = {'all': records, 'geom_core': [r for r in records if r['domain'] == 'geom_core'],
                       'charge_challenge': [r for r in records if r['domain'] == 'diverse_charge'],
                       'net_charged': [r for r in records if r['Q'] != 0], 'nH': [r for r in records if r['has_nH']]}
            for name, recs in subsets.items():
                if not recs:
                    continue
                n = len(recs)
                cm = np.array([r['q_confusion'] for r in recs]).sum(0)
                tp = cm.diagonal(); precision = tp/np.maximum(cm.sum(0), 1); recall = tp/np.maximum(cm.sum(1), 1)
                f1 = 2*precision*recall/np.maximum(precision+recall, 1e-12)
                charged_support = (cm.sum(1) > 0) & (np.arange(5) != 2)
                row = {'method': method, 'rounds': rounds, 'sigma': sigma, 'subset': name, 'n': n, 'budget': k,
                       'species_coverage': sum(any(r['species_hits']) for r in recs)/n,
                       'graph_coverage': sum(any(r['graph_hits']) for r in recs)/n,
                       'joint_label_coverage': sum(any(r['joint_hits']) for r in recs)/n,
                       'closed_shell_connected_valid_fraction': sum(r['valid_count'] for r in recs)/(n*k),
                       'valid_unique_mean': sum(r['valid_unique'] for r in recs)/n,
                       'first_h_exact': sum(r['first_h_exact'] for r in recs)/n,
                       'first_q_exact': sum(r['first_q_exact'] for r in recs)/n,
                       'first_h_atom_accuracy': sum(r['first_h_correct_atoms'] for r in recs)/sum(r['atom_n'] for r in recs),
                       'first_q_atom_accuracy': sum(r['first_q_correct_atoms'] for r in recs)/sum(r['atom_n'] for r in recs),
                       'first_charged_atom_macro_f1': float(f1[charged_support].mean()) if charged_support.any() else None,
                       'first_q_per_class': [{'q': int(q), 'support': int(cm[qi].sum()), 'precision': float(precision[qi]),
                                              'recall': float(recall[qi]), 'f1': float(f1[qi])} for qi, q in enumerate(range(-2, 3))],
                       'Q_preserved': all(r['Q_preserved'] for r in recs),
                       'decode_ms_per_input_set_all': seconds/len(rows)*1000}
                summary.append(row)
                if name in ['geom_core', 'charge_challenge']:
                    p.log(str({key: row[key] for key in ['method', 'rounds', 'sigma', 'subset', 'n', 'budget',
                                                       'species_coverage', 'graph_coverage', 'closed_shell_connected_valid_fraction',
                                                       'first_charged_atom_macro_f1', 'decode_ms_per_input_set_all']}))
            p.save_json(HERE/a.evaluation_name, {'args': vars(a), 'checkpoint_epoch': ck['epoch'],
                                               'summary': summary, 'records': allrecords,
                                               'scope': 'Developmental validation; Q provided; all methods share exact charge-sum DP; no rejection replenishment.'})


def tests():
    p.seed_all(11); RDLogger.DisableLog('rdApp.*')
    for smi in ['c1cc[nH]c1', 'C[N+](C)(C)C', '[O-]C(=O)C', '[O-][N+](=O)c1ccccc1']:
        mol = Chem.AddHs(Chem.MolFromSmiles(smi)); from rdkit.Chem import AllChem
        AllChem.EmbedMolecule(mol, randomSeed=11)
        row = make_row(mol, 1, 0, 'test')
        assert row['h'].sum() == sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms())
    logits = torch.randn(5, 25); state = torch.full((5,), AMASK)
    for Q in [[0, 1], [-4, 6], [4, -6]]:
        draw = constrained_atoms(logits, state, [2, 3], torch.tensor(Q), np.random.default_rng(11))
        assert int((draw[:2]//5-2).sum()) == Q[0] and int((draw[2:]//5-2).sum()) == Q[1]
    state[0] = 16
    draw = constrained_atoms(logits, state, [2, 3], torch.tensor([0, 1]), np.random.default_rng(11))
    assert draw[0] == state[0]
    # Exact two-atom charge distribution checked against brute-force posterior.
    l = torch.full((2, 25), -30.); l[:, 0] = 0; l[:, 10] = 1; l[:, 20] = 0
    rng = np.random.default_rng(9); seen = collections.Counter()
    for _ in range(1500):
        d = constrained_atoms(l, torch.full((2,), AMASK), [2], torch.tensor([0]), rng)
        seen[int(d[0]//5-2)] += 1
    expected_zero = math.exp(2)/(math.exp(2)+2)
    assert abs(seen[0]/1500-expected_zero) < .05, seen
    # Inference is independent of all reference target values if Z,R,Q fixed.
    row = {'feat': torch.randn(3, 360), 'base': torch.randn(3, 5), 'node_feat': torch.randn(3, 360),
           'z': torch.tensor([6, 7, 8]), 'pairs': torch.tensor([[0, 1], [1, 2], [0, 2]]),
           'labels': torch.zeros(3, dtype=torch.long), 'atom_target': torch.full((3,), 10), 'Q': 0}
    b = pack([row], 'cpu'); net = JointNet(width=32, layers=1).eval()
    for rounds in [1, 2, 4]:
        x = sample(net, b, rounds, 42)
        b['labels'].fill_(4); b['atarget'].fill_(24)
        y = sample(net, b, rounds, 42)
        assert all(torch.equal(xx, yy) for xx, yy in zip(x, y)), 'Target leakage'
    x = sample(net, b, 1, 42, atoms_after_B=True)
    b['labels'].fill_(2); b['atarget'].fill_(0)
    y = sample(net, b, 1, 42, atoms_after_B=True)
    assert all(torch.equal(xx, yy) for xx, yy in zip(x, y)), 'Block target leakage'
    ats = torch.full((3,), AMASK); bs = torch.full((3,), BMASK)
    out = net(b, bs, ats)
    swapped = dict(b, pairs=b['pairs'].flip(1))
    out2 = net(swapped, bs, ats)
    assert all(torch.allclose(x, y, atol=1e-6) for x, y in zip(out, out2)), 'Pair direction asymmetry'
    ats2 = ats.clone(); ats2[1] = 16
    # bhead starts at zero by construction; atom dependence still tested.
    assert not torch.allclose(net(b, bs, ats2)[1][0], out[1][0]), 'No adjacent atom-state dependency'
    p.log('joint tests passed: h labels, species roundtrip, DP sums/visible states/distribution, no leakage, undirected symmetry, dependency')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'train', 'evaluate', 'test'])
    parser.add_argument('--geom', default=str(p.ROOT/'data/geom_drugs_all_random1_rel10.sdf'))
    parser.add_argument('--diverse', default=str(p.ROOT/'data/diverse_subset_80k.sdf'))
    parser.add_argument('--checkpoint', default=str(p.ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    parser.add_argument('--seed', type=int, default=20261002); parser.add_argument('--device', default='cuda')
    parser.add_argument('--charge-train', type=int, default=1000); parser.add_argument('--internal-train', type=int, default=500)
    parser.add_argument('--charge-val', type=int, default=150); parser.add_argument('--internal-val', type=int, default=100)
    parser.add_argument('--noises', type=float, nargs='+', default=[0., .1, .2])
    parser.add_argument('--eval-noises', type=float, nargs='+', default=[0., .1, .2])
    parser.add_argument('--width', type=int, default=96); parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--batch', type=int, default=32); parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--lr', type=float, default=.0002); parser.add_argument('--candidates', type=int, default=8)
    parser.add_argument('--rounds', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--evaluation-name', default='evaluation.json')
    parser.add_argument('--methods', nargs='+', default=None)
    a = parser.parse_args(); torch.set_num_threads(4)
    {'prepare': prepare, 'train': train, 'evaluate': eval_run, 'test': lambda _: tests()}[a.action](a)


if __name__ == '__main__':
    main()
