"""Decompose the local-MAP gain into coupled search vs the buildability filter.

The reported +9pp (greedy -> constrained local MAP) conflates two changes at once:

  1. maximising the joint score over a COUPLED neighbourhood (several flagged
     edges move together), and
  2. a hard buildability filter that outranks the score (RDKit-sanitizable, no
     radicals, single fragment) -- greedy has no such filter at all.

Molecules that fail to build drop 72 -> 47 -> 45 going greedy -> localmap ->
+deloc, so (2) is doing something, and nothing in the published numbers says how
much. This script records, for every candidate in the beam, its score, whether it
builds, whether it is correct, and its Hamming distance from the greedy
assignment. Every selector variant is then computed offline from one run:

  greedy            candidate 0, no filter                     (published baseline)
  search_only       argmax score, filter ignored
  filter_d1         argmax (buildable, score) over distance<=1  (no coupled move)
  full              argmax (buildable, score) over the whole beam (= localmap)
  oracle_in_beam    is the reference assignment anywhere in the beam

If `filter_d1` recovers most of `full`, the mechanism is the filter and the
"coupled search" framing is wrong. If it does not, the coupling claim survives
its first real test.

Nothing here is tuned: the neighbourhood, beam and scoring are exactly the frozen
configuration, and no reference label enters any selector.
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
import constrained_map as CM
# build_reference_rows / write_dumps are shared with the main entry point on
# purpose: the locked-set protocol scores the component ladder through exactly
# the same export path as the published numbers, not through a copy of it.
import pose_localmap as PLM


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
    a.add_argument('--fraction', type=float, default=0.10)
    a.add_argument('--max_flagged', type=int, default=8)
    a.add_argument('--beam', type=int, default=48)
    a.add_argument('--topk_per_edge', type=int, default=2)
    a.add_argument('--limit', type=int, default=0)
    a.add_argument('--tag', default='')
    a.add_argument('--atom_solver', choices=['greedy', 'map'], default='greedy',
                   help="how the (q,h) state is chosen subject to sum(q)=Q. "
                        "'greedy' is joint.constrained_atoms(greedy=True), which "
                        "marginalises h, runs a SUM-product suffix recursion and "
                        "picks q position by position: it is NOT a joint MAP and "
                        "it depends on the atom order (measured: 9.2%% suboptimal "
                        "on two atoms, 14%% order flips on three). It is the "
                        "default only because every published number used it. "
                        "'map' is constrained_map.constrained_map, an exact "
                        "max-product MAP over the full 25-state vocabulary "
                        "(verified against brute force 300/300, order invariant).")
    a.add_argument('--max_dist', type=int, default=0,
                   help='0 = record all candidates (the ablation needs this)')
    a.add_argument('--reference_from_sdf', action='store_true',
                   help='take the molecule set AND the reference-edge list from '
                        '--sdf itself instead of from an archived per-edge dump. '
                        'Required for a new evaluation set (the locked test set), '
                        'which has no archived dump. Validated against the 488 dev '
                        'set, where both paths must emit identical rows.')
    a.add_argument('--population_manifest', help='fixed eligible names; no prediction-based filtering')
    a.add_argument('--names_from_baseline', action='store_true',
                   help='ignore --names and take the molecule set from '
                        '--baseline_per_edge; use this on the noise/CCD probe '
                        'conditions, which have no summary json. A switch '
                        'rather than --names "" because PowerShell drops '
                        'empty-string arguments to native commands.')
    a.add_argument('--baseline_per_edge',
                   default=str(ROOT/'results/per_edge_training_catalogue.jsonl'),
                   help='archived dump; supplies the reference-edge list so the '
                        'emitted per-edge file is scored by exactly the published '
                        'protocol')
    return a.parse_args()


def main():
    a = parse_args()
    Path(a.outdir).mkdir(parents=True, exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(11)
    # The molecule set comes from --names when that summary exists; otherwise it
    # is taken from the reference-edge dump, which is what makes this script
    # usable on the noise/CCD probe conditions (each has its own perturbed SDF
    # and its own archived per-edge dump, but no summary json).
    if a.reference_from_sdf:
        wanted = None
        if a.population_manifest:
            from population_manifest import validate_population
            refs, excluded = validate_population(a.sdf, a.population_manifest)
            wanted = set(refs)
            p.log(f'fixed population: {len(refs)} eligible, {len(excluded)} element exclusions')
        p.log(f'molecule set and reference edges taken from {a.sdf}')
    elif (not a.names_from_baseline) and a.names and Path(a.names).exists():
        wanted = {r['name'] for r in
                  json.loads(Path(a.names).read_text(encoding='utf-8'))['records']}
    else:
        wanted = set()
        for line in Path(a.baseline_per_edge).read_text(encoding='utf-8').splitlines():
            if line.strip():
                wanted.add(json.loads(line)['name'])
        p.log(f'molecule set taken from {a.baseline_per_edge}: {len(wanted)} molecules')
    rows = []
    for mol in Chem.SDMolSupplier(a.sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            if a.reference_from_sdf:
                raise ValueError('Unparseable or coordinate-free frozen reference; refusing denominator shrink')
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        if a.reference_from_sdf and name is None:
            raise ValueError('Unnamed frozen reference')
        if wanted is not None and name not in wanted:
            continue
        try:
            rows.append(J.make_row(mol, name, -1, 'pdbbind_pose'))
        except Exception as exc:
            if a.reference_from_sdf:
                raise ValueError(f'Cannot prepare frozen reference {name!r}: {exc}') from exc
            continue
        if a.limit and len(rows) >= a.limit:
            break
    p.log(f'poses: {len(rows)}')
    if a.population_manifest and not a.limit and {r['id'] for r in rows} != wanted:
        raise ValueError('Prepared rows differ from frozen eligible names')

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

    draws = np.random.default_rng(11)
    records = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for mi, r in enumerate(feats):
            b1 = J.pack([r], a.device)
            allB = torch.full_like(b1['labels'], J.BMASK)
            allA = torch.full_like(b1['atarget'], J.AMASK)
            bl, _ = net(b1, allB, allA)
            blog = bl.log_softmax(-1)
            base = blog.argmax(-1)
            conf = blog.gather(1, base[:, None]).squeeze(1)
            m = len(base)

            k = min(a.max_flagged, max(1, int(round(a.fraction*m))))
            flagged = torch.topk(-conf, k).indices.tolist()
            # neighbourhood: top-k labels per flagged edge, beam-pruned by log P(B)
            options = []
            for e in flagged:
                top = torch.topk(blog[e], a.topk_per_edge).indices.tolist()
                options.append(top)
            combos = []
            for choice in itertools.product(*options):
                score = sum(float(blog[e, c]) for e, c in zip(flagged, choice))
                combos.append((score, choice))
            combos.sort(key=lambda x: -x[0])
            combos = combos[:a.beam]

            cand = []
            for _, choice in combos:
                bvec = base.clone()
                for e, c in zip(flagged, choice):
                    bvec[e] = c
                cand.append(bvec)
            R = len(cand)
            bR = J.pack([r]*R, a.device)
            bs = torch.cat(cand)
            aR = torch.full_like(bR['atarget'], J.AMASK)
            _, al = net(bR, bs, aR)
            al = al + J.catalogue_logits(bR, bs, catalogue)
            alog = al.log_softmax(-1)
            if a.atom_solver == 'map':
                ad = torch.as_tensor(CM.constrained_map(
                    al, aR, bR['atom_lengths'], bR['Q']), device=al.device)
            else:
                ad = J.constrained_atoms(al, aR, bR['atom_lengths'], bR['Q'],
                                         draws, greedy=True)
            bsplit = bs.cpu().split(bR['lengths'])
            asplit = ad.cpu().split(bR['atom_lengths'])
            alog_split = alog.cpu().split(bR['atom_lengths'])
            blog_total = [float(blog.gather(1, c[:, None]).sum()) for c in cand]

            ref_smi = r['smiles']
            cands = []
            best = None
            bonly = None
            for t in range(R):
                smi, good = J.build_species(r, bsplit[t], asplit[t])
                atom_lp = float(alog_split[t].gather(1, asplit[t][:, None]).sum())
                score = blog_total[t] + atom_lp
                cands.append({
                    'dist': int((bsplit[t] != base.cpu()).sum()),
                    'score': score, 'good': bool(good),
                    # the two additive components of `score`, recorded separately so
                    # that the pre-registered component ladder can rank the SAME beam
                    # by edge probability alone (the conditional atom-state term is
                    # what the next rung adds). bscore = sum log P(B); ascore =
                    # log P(h,q | B) with the catalogue prior already folded in.
                    'bscore': blog_total[t], 'ascore': atom_lp,
                    'correct': bool(smi == ref_smi),
                    'exactB': bool(torch.equal(bsplit[t], r['labels']) and r['missed'] == 0),
                    'bond_labels': [int(x) for x in bsplit[t].tolist()],
                    'atom_states': [int(x) for x in asplit[t].tolist()],
                    'smiles': smi,
                })
                entry = (good, score, t, smi)
                if best is None or (entry[0], entry[1]) > (best[0], best[1]):
                    best = entry
                # pre-registered component ladder, rung B: rank the SAME beam by
                # edge probability alone, keep the same hard filter. Lexicographic
                # max on (good, bscore) is 'walk the pool in descending sum log
                # P(B) and take the first that builds'.
                bentry = (good, blog_total[t], t, smi)
                if bonly is None or (bentry[0], bentry[1]) > (bonly[0], bonly[1]):
                    bonly = bentry
            good, score, t, smi = best
            bgood, _, bt, bsmi = bonly
            # cand[0] is the per-edge argmax assignment: each edge's own top label
            # maximises its own term, so the highest-scoring combo is the greedy
            # baseline and asplit[0] is its (h,q) completion.
            assert torch.equal(cand[0], base)
            base_smi, base_good = J.build_species(r, bsplit[0], asplit[0])
            records.append({
                'name': r['id'], 'n_pairs': m, 'flagged': k, 'candidates': R,
                'z': [int(x) for x in r['z'].tolist()], 'Q': int(r['Q']),
                'selected_indices': {'greedy': 0, 'bondonly': bt, 'localmap': t},
                'baseline_species': bool(base_smi == ref_smi),
                'baseline_valid': bool(base_good),
                'baseline_strictB': bool(torch.equal(base.cpu(), r['labels']) and r['missed'] == 0),
                'localmap_species': bool(smi == ref_smi),
                'localmap_valid': bool(good),
                'localmap_strictB': bool(torch.equal(bsplit[t], r['labels']) and r['missed'] == 0),
                'changed_from_baseline': bool(not torch.equal(bsplit[t], base.cpu())),
                'reference_in_neighbourhood': bool(any(
                    torch.equal(c.cpu(), r['labels']) for c in cand)) and r['missed'] == 0,
                'pairs': [list(map(int, e)) for e in r['pairs'].tolist()],
                'localmap_labels': [int(x) for x in bsplit[t].tolist()],
                'baseline_labels': [int(x) for x in bsplit[0].tolist()],
                # component ladder rung B, recorded as its own selector so that
                # write_dumps can emit per-edge dumps for it
                'bondonly_labels': [int(x) for x in bsplit[bt].tolist()],
                'bondonly_species': bool(bsmi == ref_smi),
                'bondonly_valid': bool(bgood),
                'candidates_detail': cands,
                # the reference bond list straight from the molecule, so a new
                # evaluation set can be scored by the published protocol without
                # an archived dump to borrow the edge list from
                'ref_edges': [[int(i), int(j), int(r['z'][i]), int(r['z'][j]), int(c)]
                              for (i, j), c in sorted(r['truth'].items())],
            })
            if (mi+1) % 50 == 0:
                p.log(f'{mi+1}/{len(feats)}  {time.perf_counter()-t0:.0f}s')

    n = len(records)
    summary = {
        'n': n, 'seconds': time.perf_counter()-t0, 'args': vars(a),
        'mean_candidates': sum(r['candidates'] for r in records)/n,
        'baseline_species': sum(r['baseline_species'] for r in records)/n,
        'baseline_strictB': sum(r['baseline_strictB'] for r in records)/n,
        'baseline_valid': sum(r['baseline_valid'] for r in records)/n,
        'localmap_species': sum(r['localmap_species'] for r in records)/n,
        'localmap_strictB': sum(r['localmap_strictB'] for r in records)/n,
        'localmap_valid': sum(r['localmap_valid'] for r in records)/n,
        'bondonly_species': sum(r['bondonly_species'] for r in records)/n,
        'bondonly_valid': sum(r['bondonly_valid'] for r in records)/n,
        'changed': sum(r['changed_from_baseline'] for r in records)/n,
        'reference_in_neighbourhood': sum(r['reference_in_neighbourhood'] for r in records)/n,
        'gained': sum(r['localmap_species'] and not r['baseline_species'] for r in records),
        'lost': sum(r['baseline_species'] and not r['localmap_species'] for r in records),
    }
    print(json.dumps(summary, indent=1))
    tag = a.tag or f'f{int(a.fraction*100)}k{a.max_flagged}t{a.topk_per_edge}'
    Path(a.outdir, f'pose_localmap_{tag}.json').write_text(
        json.dumps({'summary': summary, 'records': records}, indent=1), encoding='utf-8')

    # per-edge dumps for all three decoder rungs, through the shared exporter.
    # Rung D (delocalized rewriting) is produced downstream by pose_delocalize.py
    # from the localmap dump, exactly as in the published protocol.
    base = PLM.build_reference_rows(records, a)
    stats = PLM.write_dumps(records, base, tag, a.outdir, a.sigma,
                            selectors=[('localmap_labels', 'localmap'),
                                       ('bondonly_labels', 'bondonly'),
                                       ('baseline_labels', 'greedy')])
    Path(a.outdir, f'export_stats_{tag}.json').write_text(
        json.dumps({k: {'refedge_rows': v[0], 'fullgraph_rows': v[1],
                        'off_reference_edges': v[2], 'molecules_affected': v[3]}
                    for k, v in stats.items()}, indent=1), encoding='utf-8')



if __name__ == '__main__':
    main()
