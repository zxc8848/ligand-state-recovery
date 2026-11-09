"""Scheduled (T-step) absorbing-diffusion decoding on the 488 PDBBind poses.

This is the experiment the t conditioning exists for. The earlier multi-round
confidence schedule *hurt* (``pose_iterate_summary.json``: species 0.7254 -> 0.7131
over five rounds) on a network that was never told how corrupted its input was.
Here the reverse kernel is conditioned on the realized noise level, and the number
of steps is swept, so the question becomes answerable:

    does multi-step decoding help once the kernel knows t?

T=1 reduces exactly to the deployed one-shot decode (the cosine schedule leaves no
masked edge after one step), so the T sweep is self-anchored. The atom stage is
unchanged in every setting: complete B visible, catalogue prior, exact sum(q)=Q DP.

Optionally runs the constrained local-MAP second pass on top of the final step,
with the same neighbourhood as ``pose_localmap.py`` (lowest-confidence fraction of
edges, top-2 labels each, beam-pruned, buildability as a hard filter, then the
model's own joint score), so one-shot+search and multi-step+search are comparable.
"""
import argparse
import itertools
import json
import sys
import time
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
    a.add_argument('--baseline_per_edge',
                   default=str(ROOT/'results/per_edge_training_catalogue.jsonl'))
    a.add_argument('--checkpoint', default=str(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    a.add_argument('--joint_ckpt', default=str(HERE/'joint_v4/diffusion_train/best.pt'))
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--outdir', default=str(HERE/'joint_v4/diffusion'))
    a.add_argument('--tag', default='t')
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--sigma', type=float, default=0.0)
    a.add_argument('--steps', type=int, nargs='+', default=[1, 2, 4, 8])
    a.add_argument('--candidates', type=int, default=8)
    a.add_argument('--seeds', type=int, nargs='+', default=[20261002])
    a.add_argument('--localmap', action='store_true',
                   help='also run the constrained local-MAP pass on the final step')
    a.add_argument('--fraction', type=float, default=0.10)
    a.add_argument('--max_flagged', type=int, default=8)
    a.add_argument('--topk_per_edge', type=int, default=2)
    a.add_argument('--beam', type=int, default=48)
    a.add_argument('--limit', type=int, default=0)
    return a.parse_args()


@torch.no_grad()
def schedule_decode(net, b, catalogue, steps, rng, gen=None, greedy=True):
    """T-step absorbing-diffusion decode of B, then the deployed (h,q) stage.

    Returns (B, atom states, final-step log-probs over B).
    """
    bs = torch.full_like(b['labels'], J.BMASK)
    ats = torch.full_like(b['atarget'], J.AMASK)
    blog = None
    for k in range(1, steps+1):
        bl, _ = net(b, bs, ats)
        blog = bl.log_softmax(-1)
        bp = blog.exp()
        draw = bp.argmax(-1) if greedy else torch.multinomial(bp, 1, generator=gen).squeeze(1)
        conf = bp.gather(1, draw[:, None]).squeeze(1)
        remaining = T.cosine_gamma(k/steps)
        bs = J.reveal(bs, draw, conf, b['lengths'], remaining, J.BMASK)
    assert bool((bs != J.BMASK).all()), 'schedule left masked edges'
    _, al = net(b, bs, ats)
    al = al + J.catalogue_logits(b, bs, catalogue)
    ad = J.constrained_atoms(al, ats, b['atom_lengths'], b['Q'], rng, greedy=True)
    return bs, ad, blog


@torch.no_grad()
def local_map(net, b, catalogue, bs, blog, rng, row_index, a):
    """Constrained local MAP seeded from a decoded B, one molecule at a time."""
    out = []
    estart = 0
    for g, m in enumerate(b['lengths']):
        row = row_index[g]
        base = bs[estart:estart+m]
        lg = blog[estart:estart+m]
        estart += m
        conf = lg.gather(1, base[:, None]).squeeze(1)
        k = min(a.max_flagged, max(1, int(round(a.fraction*m))))
        flagged = torch.topk(-conf, k).indices.tolist()
        options = [torch.topk(lg[e], a.topk_per_edge).indices.tolist() for e in flagged]
        combos = sorted(((sum(float(lg[e, c]) for e, c in zip(flagged, choice)), choice)
                         for choice in itertools.product(*options)), key=lambda x: -x[0])
        combos = combos[:a.beam]
        cand = []
        for _, choice in combos:
            v = base.clone()
            for e, c in zip(flagged, choice):
                v[e] = c
            cand.append(v)
        R = len(cand)
        bR = J.pack([row]*R, bs.device)
        vec = torch.cat(cand)
        aR = torch.full_like(bR['atarget'], J.AMASK)
        _, al = net(bR, vec, aR)
        al = al + J.catalogue_logits(bR, vec, catalogue)
        alog = al.log_softmax(-1)
        ad = J.constrained_atoms(al, aR, bR['atom_lengths'], bR['Q'], rng, greedy=True)
        bsp = vec.cpu().split(bR['lengths'])
        asp = ad.cpu().split(bR['atom_lengths'])
        alp = alog.cpu().split(bR['atom_lengths'])
        best = None
        for t in range(R):
            smi, good = J.build_species(row, bsp[t], asp[t])
            score = (float(lg.cpu().gather(1, cand[t].cpu()[:, None]).sum())
                     + float(alp[t].gather(1, asp[t][:, None]).sum()))
            if best is None or (good, score) > (best[0], best[1]):
                best = (good, score, t, smi)
        good, score, t, smi = best
        out.append((bsp[t], asp[t], smi, good))
    return out


def main():
    a = parse_args()
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(11)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

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
        if a.limit and len(rows) >= a.limit:
            break
    p.log(f'poses: {len(rows)}')

    backbone, bargs = p.load_backbone(a.checkpoint, a.device)
    noise = np.random.default_rng(11)
    geos = [p.geometry(r, a.sigma, noise, bargs['cutoff']) for r in rows]
    feats = []
    for i in range(0, len(geos), a.batch):
        feats.extend(J.extract(backbone, geos[i:i+a.batch], a.device))

    net, ck = T.load_joint(a.joint_ckpt, a.device)
    arch = ck.get('args', {}).get('arch', 'base')
    variant = ck.get('args', {}).get('variant', 'baseline')
    catalogue = J.make_catalogue(torch.load(a.sources, map_location='cpu',
                                           weights_only=False)['train'])
    p.log(f'checkpoint arch={arch} variant={variant} epoch={ck.get("epoch")}')

    summary = []
    per_mol = {}
    for steps in a.steps:
        rec = {}
        t0 = time.perf_counter()
        for i in range(0, len(feats), a.batch):
            chunk = feats[i:i+a.batch]
            b = J.pack(chunk, a.device)
            rng = np.random.default_rng(11+i)
            bs, ad, blog = schedule_decode(net, b, catalogue, steps, rng, greedy=True)
            bsp = bs.cpu().split(b['lengths'])
            asp = ad.cpu().split(b['atom_lengths'])
            lmap = (local_map(net, b, catalogue, bs, blog, rng, chunk, a)
                    if a.localmap else None)
            # K-candidate coverage with stochastic reveal
            cov = {}
            for s in a.seeds:
                gen = torch.Generator(device=b['feat'].device).manual_seed(s+i)
                hits = [[] for _ in chunk]
                uniq = [set() for _ in chunk]
                nvalid = [0]*len(chunk)
                for _ in range(a.candidates):
                    cb, ca, _ = schedule_decode(net, b, catalogue, steps,
                                                np.random.default_rng(s+i),
                                                gen=gen, greedy=False)
                    cbs = cb.cpu().split(b['lengths'])
                    cas = ca.cpu().split(b['atom_lengths'])
                    for g, r in enumerate(chunk):
                        smi, good = J.build_species(r, cbs[g], cas[g])
                        hits[g].append(smi == r['smiles'])
                        nvalid[g] += bool(good)
                        if good:
                            uniq[g].add(smi)
                cov[s] = (hits, nvalid, uniq)
            for g, r in enumerate(chunk):
                smi, good = J.build_species(r, bsp[g], asp[g])
                e = {'greedy_species': bool(smi == r['smiles']),
                     'greedy_valid': bool(good),
                     'greedy_strictB': bool(torch.equal(bsp[g], r['labels'])
                                            and r['missed'] == 0),
                     'labels': [int(x) for x in bsp[g].tolist()],
                     'pairs': [list(map(int, x)) for x in r['pairs'].tolist()]}
                for s in a.seeds:
                    hits, nvalid, uniq = cov[s]
                    e[f'cov{a.candidates}_{s}'] = bool(any(hits[g]))
                    e[f'top1_{s}'] = bool(hits[g][0])
                    e[f'validfrac_{s}'] = nvalid[g]/a.candidates
                    e[f'uniq_{s}'] = len(uniq[g])
                if lmap is not None:
                    lb, la, lsmi, lgood = lmap[g]
                    e['localmap_species'] = bool(lsmi == r['smiles'])
                    e['localmap_valid'] = bool(lgood)
                    e['localmap_strictB'] = bool(torch.equal(lb, r['labels'])
                                                 and r['missed'] == 0)
                    e['localmap_labels'] = [int(x) for x in lb.tolist()]
                rec[r['id']] = e
            p.log(f'T={steps}: {min(i+a.batch, len(feats))}/{len(feats)}')
        n = len(rec)
        row = {'steps': steps, 'n': n, 'seconds': time.perf_counter()-t0,
               'greedy_species': sum(v['greedy_species'] for v in rec.values())/n,
               'greedy_strictB': sum(v['greedy_strictB'] for v in rec.values())/n,
               'greedy_valid': sum(v['greedy_valid'] for v in rec.values())/n,
               'coverage_mean': float(np.mean([[v[f'cov{a.candidates}_{s}'] for s in a.seeds]
                                               for v in rec.values()])),
               'sampled_top1_mean': float(np.mean([[v[f'top1_{s}'] for s in a.seeds]
                                                   for v in rec.values()])),
               'valid_fraction_mean': float(np.mean([[v[f'validfrac_{s}'] for s in a.seeds]
                                                     for v in rec.values()])),
               'unique_species_mean': float(np.mean([[v[f'uniq_{s}'] for s in a.seeds]
                                                     for v in rec.values()]))}
        if a.localmap:
            row.update({
                'localmap_species': sum(v['localmap_species'] for v in rec.values())/n,
                'localmap_strictB': sum(v['localmap_strictB'] for v in rec.values())/n,
                'localmap_valid': sum(v['localmap_valid'] for v in rec.values())/n})
        summary.append(row)
        per_mol[steps] = rec
        p.log(json.dumps(row))

    # per-edge dumps in the archived format, one per step count (and per selector)
    base = {}
    for line in Path(a.baseline_per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            base.setdefault(r['name'], []).append(r)
    for steps, rec in per_mol.items():
        keys = [('labels', 'greedy')] + ([('localmap_labels', 'localmap')] if a.localmap else [])
        for key, sel in keys:
            lines = []
            for name, e in rec.items():
                pm = {tuple(sorted(x)): c for x, c in zip([tuple(y) for y in e['pairs']], e[key])}
                for orow in base.get(name, []):
                    edge = (orow['i'], orow['j'])
                    pred = pm.get(tuple(sorted(edge)), 0)
                    lines.append(json.dumps({
                        'name': name, 'sigma': a.sigma, 'i': edge[0], 'j': edge[1],
                        'z_i': orow['z_i'], 'z_j': orow['z_j'], 'true': orow['true'],
                        'pred': pred, 'correct': int(pred == orow['true'])}))
            path = out/f'pose_{sel}_{a.tag}_T{steps}_refedges.jsonl'
            path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            p.log(f'wrote {path}')

    (out/f'pose_diffusion_{a.tag}.json').write_text(json.dumps(
        {'args': vars(a), 'arch': arch, 'variant': variant,
         'checkpoint_epoch': ck.get('epoch'), 'summary': summary,
         'records': {str(k): v for k, v in per_mol.items()}}, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
