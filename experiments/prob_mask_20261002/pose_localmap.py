"""Constrained local MAP decoding: the "second pass" done as a coupled move.

Measured facts this is built on (``pose_confidence.json``, ``pose_headroom.json``,
``pose_iterate_summary.json`` on the 488 PDBBind poses):

  * if B were correct, the deployed (h,q) + catalogue + charge-DP stage recovers
    96.7% of species -- essentially all of the residual error is in B;
  * flagging the lowest-confidence 10% of a molecule's candidate edges (~5.9
    edges) contains every wrong edge in 89.3% of molecules, so the errors are
    easy to localise;
  * but 112 of the 176 imperfect molecules have two or more wrong edges, and
    re-predicting flagged edges one at a time *lowers* species recovery
    (0.7254 -> 0.7131 after five confidence rounds), because a single/double
    swap inside one group has to move both edges at once.

So the refinement unit here is a *joint assignment over the flagged edge set*,
scored as a complete graph under the model's own factorisation

    log P(B) + log P(h,q | B)  (+ catalogue prior, exact sum(q)=Q)

with chemical buildability as a hard filter. The search neighbourhood is the
top-2 marginal labels of each flagged edge, beam-pruned by the B term. Nothing
in the objective or the filter consults a reference label.
"""
import argparse
import itertools
import json
import random
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
    a.add_argument('--reference_from_sdf', action='store_true',
                   help='take the molecule set AND the reference-edge list from '
                        '--sdf itself instead of from an archived per-edge dump. '
                        'Required for a new evaluation set (the locked test set), '
                        'which has no archived dump. Validated against the 488 dev '
                        'set, where both paths must emit identical rows.')
    a.add_argument('--population_manifest', help='fixed eligible names; no prediction-based filtering')
    a.add_argument('--noise_seed', type=int, default=11,
                   help='seed of the coordinate-noise stream used when --sigma > 0. '
                        'Every published condition reads pre-perturbed coordinates '
                        'with --sigma 0, so this stream is inert for them and 11 '
                        'reproduces them. Varying it measures the noise-sampling '
                        'component of variance, which the three training seeds do '
                        'not contain.')
    a.add_argument('--permute_seed', type=int, default=0,
                   help='0 (default) keeps the SDF atom order and reproduces every '
                        'released number. A non-zero value randomly renumbers each '
                        "molecule's atoms before featurisation, records the "
                        'permutation, and emits predictions mapped BACK to the '
                        'original indices. This exists only to test the atom-order '
                        'dependence of the default charge solver '
                        '(joint.constrained_atoms with greedy=True); it is a '
                        'diagnostic path and must stay 0 for any reported result.')
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
    permutations = {}
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
            if a.permute_seed:
                # Chem.RenumberAtoms(mol, order): new position i holds old atom
                # order[i], so old_index = order[new_index] is the inverse map we
                # store for remapping predictions back before comparison.
                order = list(range(mol.GetNumAtoms()))
                random.Random((a.permute_seed, name)).shuffle(order)
                mol = Chem.RenumberAtoms(mol, order)
                permutations[name] = order
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
    rng = np.random.default_rng(a.noise_seed)
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

            best = None
            for t in range(R):
                smi, good = J.build_species(r, bsplit[t], asplit[t])
                atom_lp = float(alog_split[t].gather(1, asplit[t][:, None]).sum())
                score = blog_total[t] + atom_lp
                entry = (good, score, t, smi)
                if best is None or (entry[0], entry[1]) > (best[0], best[1]):
                    best = entry
            good, score, t, smi = best
            ref_smi = r['smiles']
            # cand[0] is the per-edge argmax assignment: each edge's own top label
            # maximises its own term, so the highest-scoring combo is the greedy
            # baseline and asplit[0] is its (h,q) completion.
            assert torch.equal(cand[0], base)
            base_smi, base_good = J.build_species(r, bsplit[0], asplit[0])
            records.append({
                'name': r['id'], 'n_pairs': m, 'flagged': k, 'candidates': R,
                'permutation': permutations.get(r['id']),
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
                # The selected joint (q,h) state per atom, and the built SMILES.
                # Recorded so that atom_order_invariance.py can tell apart two
                # outcomes a SMILES comparison conflates: the same molecule, and
                # the same molecule with q/h placed on different atoms.
                # State index s encodes q = s//5 - 2 and h = s%5.
                'localmap_states': [int(x) for x in asplit[t].tolist()],
                'localmap_smiles': smi,
                # the reference bond list straight from the molecule, so a new
                # evaluation set can be scored by the published protocol without
                # an archived dump to borrow the edge list from
                'ref_edges': [[int(i), int(j), int(r['z'][i]), int(r['z'][j]), int(c)]
                              for (i, j), c in sorted(r['truth'].items())],
                'z': [int(x) for x in r['z'].tolist()],
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
        'changed': sum(r['changed_from_baseline'] for r in records)/n,
        'reference_in_neighbourhood': sum(r['reference_in_neighbourhood'] for r in records)/n,
        'gained': sum(r['localmap_species'] and not r['baseline_species'] for r in records),
        'lost': sum(r['baseline_species'] and not r['localmap_species'] for r in records),
    }
    print(json.dumps(summary, indent=1))
    tag = a.tag or f'f{int(a.fraction*100)}k{a.max_flagged}t{a.topk_per_edge}'
    Path(a.outdir, f'pose_localmap_{tag}.json').write_text(
        json.dumps({'summary': summary, 'records': records}, indent=1), encoding='utf-8')

    base = build_reference_rows(records, a)
    write_dumps(records, base, tag, a.outdir, a.sigma)


def build_reference_rows(records, a):
    """{name: [reference-edge rows]} -- from the archived dump, or from the
    molecules themselves when --reference_from_sdf."""
    if a.reference_from_sdf:
        base = {r['name']: [{'name': r['name'], 'i': i, 'j': j, 'z_i': zi,
                             'z_j': zj, 'true': c}
                            for i, j, zi, zj, c in r['ref_edges']] for r in records}
        p.log(f'reference edges from the molecules themselves: '
              f'{sum(len(v) for v in base.values())} rows')
        return base
    base = {}
    for line in Path(a.baseline_per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            base.setdefault(row['name'], []).append(row)
    return base


SELECTOR_KEYS = [('localmap_labels', 'localmap'), ('baseline_labels', 'greedy')]


def write_dumps(records, base, tag, outdir, sigma, log=None, selectors=None):
    """TWO dumps per selector, because they answer different questions.

      *_refedges.jsonl   predictions projected onto the reference edge list.
                         The published protocol, and the only thing comparable
                         with the archived numbers -- but BLIND to predicted
                         bonds on pairs the reference does not bond, so a false
                         positive there costs nothing.
      *_fullgraph.jsonl  reference edges UNION every pair the model predicted
                         non-zero. The honest full-graph evaluation, and the one
                         that should be primary.

    evaluate_species_recovery.py builds its predicted molecule from any row with
    pred > 0 regardless of `true`, so the extra rows are scored with no change to
    the scorer.

    Found 2026-10-04 by external review. Measured effect on the published pose
    numbers: 0 molecules at sigma=0, 1 at sigma=0.05, 1 at 0.10, 1-2 at 0.15 --
    so the historical numbers barely move. The interface was wrong regardless,
    and --reference_from_sdf (written for the locked test set, where there is no
    archived dump to borrow an edge list from) would have dropped every
    off-reference predicted bond.

    `selectors` is [(record key holding the label vector, dump name)]; it
    defaults to the two the published protocol uses. pose_ablate_search.py passes
    the pre-registered component ladder's extra rungs through it, so that every
    rung is dumped by this one function rather than by a copy of it.

    Returns {selector: (n_refedge_rows, n_fullgraph_rows, n_off_reference_edges,
    n_molecules_affected)}.
    """
    say = log if log is not None else p.log
    byname = {r['name']: r for r in records}
    stats = {}
    for key, suffix in (selectors or SELECTOR_KEYS):
        ref_lines, full_lines, extra_total, extra_mols = [], [], 0, 0
        for name, rec in byname.items():
            pm = dict(zip([tuple(sorted(x)) for x in rec['pairs']], rec[key]))
            zs = rec['z']
            seen = set()
            for orow in base.get(name, []):
                e = (orow['i'], orow['j'])
                k = tuple(sorted(e))
                seen.add(k)
                row = {'name': name, 'sigma': sigma, 'i': e[0], 'j': e[1],
                       'z_i': orow['z_i'], 'z_j': orow['z_j'], 'true': orow['true'],
                       'pred': pm.get(k, 0)}
                row['correct'] = int(row['pred'] == row['true'])
                ref_lines.append(json.dumps(row))
                full_lines.append(json.dumps(row))
            # pairs the model bonded that the reference edge list never mentions:
            # false positives, true = 0 by construction
            extra = sorted((k, c) for k, c in pm.items() if c != 0 and k not in seen)
            if extra:
                extra_mols += 1
                extra_total += len(extra)
            for (i, j), c in extra:
                full_lines.append(json.dumps({
                    'name': name, 'sigma': sigma, 'i': i, 'j': j,
                    'z_i': zs[i], 'z_j': zs[j], 'true': 0,
                    'pred': int(c), 'correct': 0, 'off_reference': True}))
        for lines, kind in ((ref_lines, 'refedges'), (full_lines, 'fullgraph')):
            path = Path(outdir, f'pose_{suffix}_{tag}_{kind}.jsonl')
            path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            say(f'wrote {path} ({len(lines)} rows)')
        say(f'  {suffix}: {extra_total} off-reference predicted bonds in '
            f'{extra_mols} molecules were invisible to the refedges dump')
        stats[suffix] = (len(ref_lines), len(full_lines), extra_total, extra_mols)
    return stats


def _selftest():
    """Deterministic check that an off-reference predicted bond reaches the
    fullgraph dump and not the refedges dump. Real conditions contain 0-2 such
    edges per 488 molecules, which is far too few to rely on as a test.
    """
    import tempfile
    rec = {'name': 'synthetic', 'z': [6, 6, 8, 8],
           'pairs': [[0, 1], [1, 2], [0, 3], [2, 3]],
           'localmap_labels': [1, 2, 1, 1],   # 2-3 is predicted but not a reference bond
           'baseline_labels': [1, 2, 1, 0],
           'ref_edges': [[0, 1, 6, 6, 1], [1, 2, 6, 8, 2], [0, 3, 6, 8, 1]]}
    base = {'synthetic': [{'name': 'synthetic', 'i': i, 'j': j, 'z_i': zi, 'z_j': zj,
                           'true': c} for i, j, zi, zj, c in rec['ref_edges']]}
    with tempfile.TemporaryDirectory() as d:
        stats = write_dumps([rec], base, 't', d, 0.0, log=lambda *_: None)
        ref = [json.loads(x) for x in
               Path(d, 'pose_localmap_t_refedges.jsonl').read_text().splitlines() if x.strip()]
        full = [json.loads(x) for x in
                Path(d, 'pose_localmap_t_fullgraph.jsonl').read_text().splitlines() if x.strip()]
        assert len(ref) == 3, ref
        assert len(full) == 4, full
        off = [r for r in full if r.get('off_reference')]
        assert len(off) == 1 and (off[0]['i'], off[0]['j']) == (2, 3), off
        assert off[0]['true'] == 0 and off[0]['pred'] == 1 and off[0]['correct'] == 0
        assert off[0]['z_i'] == 8 and off[0]['z_j'] == 8
        assert stats['localmap'] == (3, 4, 1, 1), stats['localmap']
        # greedy predicted 0 on that pair, so it has no off-reference row
        gfull = [json.loads(x) for x in
                 Path(d, 'pose_greedy_t_fullgraph.jsonl').read_text().splitlines() if x.strip()]
        assert len(gfull) == 3 and not any(r.get('off_reference') for r in gfull)
        assert stats['greedy'] == (3, 3, 0, 0), stats['greedy']
    print('pose_localmap._selftest: off-reference bond reaches fullgraph only  OK')


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        _selftest()
    else:
        main()
