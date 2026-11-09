"""Development-only missing-net-charge pilot for the current single-stage model."""
import argparse
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, RDLogger, rdBase
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP = ROOT / 'experiments/prob_mask_20261002'
sys.path.insert(0, str(EXP))
import probe as P
import joint as J
import jointt as T
from evaluation_common import reference_population, score_dump

ENCODER = ROOT / 'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'
CACHE = EXP / 'joint_v1/features.pt'
SOURCES = EXP / 'joint_v1/sources.pt'
DATA = {'pose': ROOT / 'work/paper2_noise_ccd_probe/pdbbind_sigma0p00/reference_for_scoring.sdf',
        'ccd_dev': ROOT / 'experiments/atom_state_candidates_20261102/development.sdf'}
N = {'pose': 488, 'ccd_dev': 332}
MASKS = (-2, -1, 0, 1, 2)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def frozen_head(seed):
    return EXP / f'joint_v5/seed{seed}/best.pt'


@torch.no_grad()
def val(net, rows, a):
    # All-MASK objective. Each molecule has equal weight within its Q stratum;
    # neutral/charged have equal weight for the atom part of epoch selection.
    be, atoms = [], {False: [], True: []}
    for start in range(0, len(rows), a.batch):
        rs = rows[start:start+a.batch]
        b = J.pack(rs, a.device)
        b['Q'].zero_()
        bl, al = net(b, torch.full_like(b['labels'], J.BMASK),
                     torch.full_like(b['atarget'], J.AMASK))
        be += [float(x.mean()) for x in F.cross_entropy(bl, b['labels'], reduction='none').split(b['lengths']) if len(x)]
        for r, loss in zip(rs, F.cross_entropy(al, b['atarget'], reduction='none').split(b['atom_lengths'])):
            atoms[r['Q'] != 0].append(float(loss.mean()))
    return dict(bond=float(np.mean(be)), atom_q0=float(np.mean(atoms[False])),
                atom_nonzero=float(np.mean(atoms[True])))


def train(a, out):
    head = frozen_head(a.seed)
    net, ck = T.load_joint(head, a.device)
    cache = torch.load(CACHE, map_location='cpu', weights_only=False)
    if cache['metadata']['checkpoint_sha256'] != sha(ENCODER):
        raise ValueError('Frozen feature cache was extracted by another encoder')
    data = cache['rows']
    net.qembed.requires_grad_(False)  # Fixed, constant Q=0 representation; no reference Q input.
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=a.lr, weight_decay=1e-4)
    rng = np.random.default_rng(a.seed)
    history, best = [], float('inf')
    save(out/'provenance.json', {'seed': a.seed, 'head_sha256': sha(head), 'encoder_sha256': sha(ENCODER),
         'cache_sha256': sha(CACHE), 'split_sizes': {k: len(v) for k, v in data.items()},
         'q_embedding': 'fixed constant zero input, frozen embedding parameters', 'epochs': a.epochs,
         'lr': a.lr, 'batch': a.batch})
    t0 = time.perf_counter()
    for epoch in range(a.epochs+1):
        net.train(bool(epoch))
        losses = []
        if epoch:
            order = rng.permutation(len(data['train']))
            for j in range(0, len(order), a.batch):
                b = J.pack([data['train'][i] for i in order[j:j+a.batch]], a.device)
                b['Q'].zero_()
                bs, ats, bsup, asup = J.corrupt_states(b)
                bl, al = net(b, bs, ats)
                loss = F.cross_entropy(bl[bsup], b['labels'][bsup]) + F.cross_entropy(al[asup], b['atarget'][asup])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                opt.step()
                losses.append(float(loss.detach()))
                if (len(losses) % 100) == 0:
                    print(f'epoch {epoch}: {j+a.batch}/{len(order)} loss {np.mean(losses[-100:]):.4f}', flush=True)
        net.eval()
        v = val(net, data['val'], a)
        score = v['bond'] + .5 * (v['atom_q0'] + v['atom_nonzero'])
        row = dict(epoch=epoch, train_loss=float(np.mean(losses)) if losses else None, **v,
                   val_objective=score, seconds=time.perf_counter()-t0)
        history.append(row)
        print(row, flush=True)
        save(out/'training.json', history)
        if epoch and score < best:
            best = score
            torch.save({'state': net.state_dict(), 'args': ck['args'], 'epoch': epoch,
                        'q_mode': 'blind_zero_embedding', 'val_objective': score}, out/'best.pt')


def top_by_charge(lp):
    """Best joint assignment per total Q, same logits for every charge hypothesis.

    lp is (beam, atoms, 25); intermediate sums are not pruned. Returns
    (beam, 5, atoms), with final Q in {-2,-1,0,1,2}. No reference Q is read.
    """
    r, n, _ = lp.shape
    local = lp.reshape(r, n, 5, 5)
    h = local.argmax(axis=-1)
    local = local.max(axis=-1)
    dp = np.zeros((r, 1), dtype=lp.dtype)
    pointers = []
    for i in range(n):
        width = dp.shape[1]
        take = np.full((r, width+4, 5), -np.inf, dtype=lp.dtype)
        for q in range(5):
            take[:, q:q+width, q] = dp + local[:, i, q, None]
        pointers.append(take.argmax(axis=-1))
        dp = take.max(axis=-1)
    qi = np.broadcast_to(np.asarray(MASKS)[None, :] + 2*n, (r, len(MASKS))).copy()
    assignment = np.empty((r, len(MASKS), n), dtype=np.int64)
    ri = np.arange(r)[:, None]
    for i in reversed(range(n)):
        q = pointers[i][ri, qi]
        assignment[:, :, i] = 5*q + h[ri, i, q]
        qi -= q
    assert (qi == 0).all()
    assert ((assignment//5-2).sum(axis=-1) == MASKS).all()
    return assignment


@torch.no_grad()
def predict(net, r, catalogue, device, supplied_q, modes):
    # r has geometry/features and only dummy label tensors. Reference Q is passed
    # explicitly ONLY in the known-control call. Unknown modes share Q=0 logits.
    row = dict(r, Q=0 if supplied_q is None else supplied_q)
    b = J.pack([row], device)
    bl, _ = net(b, torch.full_like(b['labels'], J.BMASK), torch.full_like(b['atarget'], J.AMASK))
    blog = bl.log_softmax(-1)
    conf, base = blog.max(dim=-1)
    k = min(len(base), 8, max(1, round(.10*len(base))))
    flagged = torch.topk(-conf, k).indices
    options = torch.topk(blog[flagged], 2, dim=-1).indices.cpu().tolist()
    edge = blog[flagged].cpu()
    ranked = sorted(itertools.product(*options), key=lambda c: -sum(float(edge[i, x]) for i,x in enumerate(c)))[:48]
    bc = base.repeat(len(ranked), 1)
    if k:
        bc[:, flagged] = torch.as_tensor(ranked, device=device)
    assert torch.equal(bc[0], base)
    bscore = blog.expand(len(ranked), -1, -1).gather(-1, bc[..., None]).sum(1).cpu().flatten().numpy()
    pack = J.pack([row]*len(ranked), device)
    bs = bc.flatten()
    am = torch.full_like(pack['atarget'], J.AMASK)
    _, al = net(pack, bs, am)
    al += J.catalogue_logits(pack, bs, catalogue)
    logp = al.log_softmax(-1).cpu().numpy().reshape(len(ranked), len(row['z']), 25)
    bc = bc.cpu()
    states = {}
    if 'fixed' in modes:
        ad = J.constrained_atoms(al, am, pack['atom_lengths'], pack['Q'], np.random.default_rng(11), greedy=True)
        states['fixed'] = ad.cpu().numpy().reshape(len(ranked), 1, -1)
    if 'free' in modes:
        states['free'] = logp.argmax(-1)[:, None, :]
    if 'multi_q' in modes:
        states['multi_q'] = top_by_charge(logp)
    result = {}
    for mode, aa in states.items():
        # Score all states with the SAME normalized logits. No conditioning on
        # the enumerated charge, no Q oracle, and no reference-based reranking.
        ascore = np.take_along_axis(logp[:, None, :, :], aa[..., None], axis=-1).squeeze(-1).sum(-1)
        scores = bscore[:, None] + ascore
        order = np.argsort(-scores.ravel(), kind='stable')
        best = None
        for flat in order:
            i, j = np.unravel_index(flat, scores.shape)
            state = torch.from_numpy(aa[i,j])
            smi, good = J.build_species(row, bc[i], state)
            if best is None or good:
                best = dict(smiles=smi, buildable=bool(good), B=bc[i].tolist(), states=state.tolist())
            if good:
                break
        result[mode] = best
    return result


def metrics(rows):
    def count(key):
        return sum(int(r[key]) for r in rows)
    n = len(rows)
    if not n:
        return {'n': 0}
    y = np.concatenate([r['q_ref'] for r in rows]); p = np.concatenate([r['q_pred'] for r in rows])
    # Fixed true-support classes within the cohort, shared across all systems.
    f1 = [2*((y == c) & (p == c)).sum()/((y == c).sum()+(p == c).sum()) for c in np.unique(y)]
    qok = count('net_correct')
    return dict(n=n, **{k: count(k)/n for k in ('native', 'buildable', 'strict', 'resonance', 'graph_smiles', 'net_correct', 'q_exact')},
                net_correct_count=qok, native_when_net_correct=count('native')/qok if qok else None,
                q_macro_f1=float(np.mean(f1)), q_classes=np.unique(y).tolist())


def evaluate(a, out):
    head = frozen_head(a.seed)
    net, _ = T.load_joint(head, a.device)
    new = T.load_joint(a.retrained, a.device)[0] if a.retrained else None
    encoder, cfg = P.load_backbone(ENCODER, a.device)
    cat = J.make_catalogue(torch.load(SOURCES, map_location='cpu', weights_only=False)['train'])
    for name in a.datasets:
        path = DATA[name]
        refs, skipped = reference_population(path)
        if len(refs) != N[name] or skipped:
            raise ValueError(f'Unexpected {name} population: {len(refs)}, excluded={skipped}')
        geos, target = [], {}
        rng = np.random.default_rng(11)
        for key, mol in refs.items():
            r = J.make_row(mol, key, -1, name)
            target[key] = r
            geo = P.geometry(dict(id=key, z=r['z'], coord=r['coord'], truth={}), 0., rng, cfg['cutoff'])
            geo.pop('truth')
            geo['atom_target'] = torch.zeros_like(r['z'])
            geos.append(geo)
        feats = []
        for j in range(0, len(geos), a.batch):
            feats.extend(J.extract(encoder, geos[j:j+a.batch], a.device))
        predictions = {}
        t0 = time.perf_counter()
        for i, r in enumerate(feats, 1):
            ref = target[r['id']]
            preds = {f'frozen_{k}': v for k,v in predict(net, r, cat, a.device, None, ('fixed','free','multi_q')).items()}
            preds['known_q'] = preds['frozen_fixed'] if ref['Q'] == 0 else predict(net, r, cat, a.device, ref['Q'], ('fixed',))['fixed']
            if new is not None:
                preds.update({f'trained_{k}': v for k,v in predict(new, r, cat, a.device, None, ('free','multi_q')).items()})
            for mode, pred in preds.items():
                q = [s//5-2 for s in pred['states']]
                q_ref = ref['q'].tolist()
                netq = sum(q)
                native = bool(pred['buildable'] and pred['smiles'] == ref['smiles'])
                if native and netq != ref['Q']:
                    raise AssertionError('Native identity matched with the wrong net charge')
                row = dict(name=r['id'], q_ref=q_ref, q_pred=q, net_ref=ref['Q'], net_pred=netq,
                           q_exact=(q == q_ref), native=native, net_correct=(netq == ref['Q']), **pred)
                predictions.setdefault(mode, []).append(row)
            if i % 50 == 0:
                print(f'{name}: {i}/{len(feats)} {time.perf_counter()-t0:.1f}s', flush=True)
        featmap = {r['id']: r for r in feats}
        save(out/f'{name}_meta.json', {'seed': a.seed, 'reference_sha256': sha(path), 'encoder_sha256': sha(ENCODER),
             'frozen_head_sha256': sha(head), 'retrained_head_sha256': sha(a.retrained) if a.retrained else None,
             'catalogue_source_sha256': sha(SOURCES),
             'rdkit': rdBase.rdkitVersion, 'elapsed_s': time.perf_counter()-t0})
        for mode, rows in predictions.items():
            edge_file = out/f'{name}_{mode}.jsonl'
            with edge_file.open('w', encoding='utf-8') as f:
                for row in rows:
                    key = row['name']; truth = target[key]['truth']
                    bonds = {tuple(e): c for e,c in zip(featmap[key]['pairs'].tolist(), row['B']) if c}
                    for i,j in sorted(set(bonds) | set(truth)):
                        f.write(json.dumps({'name':key, 'i':i, 'j':j, 'true':truth.get((i,j),0), 'pred':bonds.get((i,j),0)})+'\n')
            verdict = {r['name']:r for r in score_dump(edge_file, path, expected_n=N[name])['per_molecule']}
            for r in rows:
                v = verdict[r['name']]
                r.update(strict=v['strict'], resonance=v['unified_resonance'], graph_smiles=v['smiles_species'])
            groups = {'all':rows, 'net_neutral':[r for r in rows if r['net_ref']==0],
                      'net_charged':[r for r in rows if r['net_ref']!=0],
                      'cation':[r for r in rows if r['net_ref']>0], 'anion':[r for r in rows if r['net_ref']<0],
                      'internal_charge':[r for r in rows if r['net_ref']==0 and any(r['q_ref'])]}
            summary = {k:metrics(v) for k,v in groups.items()}
            save(out/f'{name}_{mode}.json', dict(summary=summary, records=rows))
            print(name, mode, json.dumps({k:{m:v.get(m) for m in ('n','native','net_correct','buildable')} for k,v in summary.items()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['train','evaluate'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--retrained', type=Path)
    p.add_argument('--seed', type=int, default=20261002, choices=[20261002,20261003,20261004])
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--datasets', nargs='+', choices=list(DATA), default=list(DATA))
    a = p.parse_args()
    torch.set_num_threads(4)
    RDLogger.DisableLog('rdApp.*')
    P.seed_all(a.seed)
    a.out.mkdir(parents=True, exist_ok=False)
    (train if a.action == 'train' else evaluate)(a, a.out)


if __name__ == '__main__':
    main()
