"""Shared local-search candidate generation and reference-blind candidate features.

Used by ``pose_localmap.py`` (model-score selection) and by ``verifier.py``
(learned selection + conformal sets). The neighbourhood is the one measured in
``results/MASK_REFINEMENT_DESIGN_2026-10-03.md``: flag the lowest-confidence
fraction of a molecule's candidate edges after the one-shot all-MASK pass, take
each flagged edge's top-k marginal labels, beam-prune the joint assignments by
log P(B), complete (h,q) for every surviving assignment with the catalogue prior
and the exact sum(q)=Q DP.

No function here reads a reference label; `label_candidates` does, and is used
only to build verifier training targets.
"""
import itertools

import numpy as np
import torch
from rdkit import Chem

import joint as J

FEATURE_NAMES = [
    'logPB_sum', 'logPB_mean', 'logPB_min', 'logPB_flagged_sum',
    'logPA_sum', 'logPA_mean', 'logPA_min',
    'cat_sum', 'cat_min', 'cat_fallback_frac',
    'n_changed', 'builds', 'closed_shell', 'connected',
    'n_radicals', 'n_fragments', 'abs_charge_sum', 'max_abs_charge',
    'frac_aromatic', 'frac_double', 'frac_triple', 'frac_nobond',
    'n_atoms', 'n_pairs', 'max_order_sum', 'mean_order_sum',
]


def _catalogue_fallback_fraction(z, pairs, labels, catalogue):
    """Fraction of atoms whose exact local-environment key is unseen in training."""
    exact, coarse, _ = catalogue
    neighbours = [[] for _ in z]
    for (i, j), c in zip(pairs, labels):
        if c:
            neighbours[i].append((z[j], c))
            neighbours[j].append((z[i], c))
    miss = 0
    for zi, nei in zip(z, neighbours):
        if (zi, tuple(sorted(nei))) not in exact:
            miss += 1
    return miss/max(len(z), 1)


@torch.no_grad()
def propose(net, row, catalogue, device, fraction=0.10, max_flagged=8,
            topk_per_edge=2, beam=48):
    """Return (candidates, info) for one molecule.

    candidates: list of dicts with 'B' (LongTensor over pairs), 'A' (atom states),
    'smiles', 'valid', 'features' (np.float32 vector, reference-blind).
    info: {'flagged': k, 'n_pairs': m, 'greedy_index': 0}
    """
    b1 = J.pack([row], device)
    allB = torch.full_like(b1['labels'], J.BMASK)
    allA = torch.full_like(b1['atarget'], J.AMASK)
    bl, _ = net(b1, allB, allA)
    blog = bl.log_softmax(-1)
    base = blog.argmax(-1)
    conf = blog.gather(1, base[:, None]).squeeze(1)
    m = len(base)

    k = min(max_flagged, max(1, int(round(fraction*m))))
    flagged = torch.topk(-conf, k).indices.tolist()
    options = [torch.topk(blog[e], topk_per_edge).indices.tolist() for e in flagged]
    combos = []
    for choice in itertools.product(*options):
        s = sum(float(blog[e, c]) for e, c in zip(flagged, choice))
        combos.append((s, choice))
    combos.sort(key=lambda x: -x[0])
    combos = combos[:beam]

    vectors = []
    for _, choice in combos:
        v = base.clone()
        for e, c in zip(flagged, choice):
            v[e] = c
        vectors.append(v)
    R = len(vectors)

    bR = J.pack([row]*R, device)
    bs = torch.cat(vectors)
    aR = torch.full_like(bR['atarget'], J.AMASK)
    _, al_raw = net(bR, bs, aR)
    prior = J.catalogue_logits(bR, bs, catalogue)
    al = al_raw + prior
    alog = al.log_softmax(-1)
    priorlog = prior.log_softmax(-1)
    rng = np.random.default_rng(0)
    ad = J.constrained_atoms(al, aR, bR['atom_lengths'], bR['Q'], rng, greedy=True)

    bsplit = bs.cpu().split(bR['lengths'])
    asplit = ad.cpu().split(bR['atom_lengths'])
    alog_split = alog.cpu().split(bR['atom_lengths'])
    prior_split = priorlog.cpu().split(bR['atom_lengths'])

    flagged_t = torch.tensor(flagged, dtype=torch.long)
    n_atoms = len(row['z'])
    zlist = row['z'].tolist()
    pairlist = row['pairs'].tolist()
    orders = torch.tensor([0., 1., 2., 3., 1.5, 0.])
    out = []
    for t in range(R):
        B = bsplit[t]
        A = asplit[t]
        lpb = blog.cpu().gather(1, B[:, None]).squeeze(1)
        lpa = alog_split[t].gather(1, A[:, None]).squeeze(1)
        cat = prior_split[t].gather(1, A[:, None]).squeeze(1)
        smi, valid = J.build_species(row, B, A)
        # chemistry counters from the raw construction (no sanitization needed)
        order_sum = torch.zeros(n_atoms)
        for (i, j), c in zip(row['pairs'].tolist(), B.tolist()):
            if c:
                order_sum[i] += orders[c]
                order_sum[j] += orders[c]
        q = A//5-2
        n_frag = 0
        n_rad = 0
        if smi is not None:
            try:
                built = Chem.MolFromSmiles(smi, sanitize=True)
                if built is not None:
                    n_frag = len(Chem.GetMolFrags(built))
                    n_rad = sum(at.GetNumRadicalElectrons() for at in built.GetAtoms())
            except Exception:
                pass
        feats = np.array([
            float(lpb.sum()), float(lpb.mean()), float(lpb.min()),
            float(lpb[flagged_t].sum()),
            float(lpa.sum()), float(lpa.mean()), float(lpa.min()),
            float(cat.sum()), float(cat.min()),
            _catalogue_fallback_fraction(zlist, pairlist, B.tolist(), catalogue),
            float((B != base.cpu()).sum()),
            float(smi is not None), float(valid), float(n_frag == 1),
            float(n_rad), float(n_frag),
            float(q.abs().sum()), float(q.abs().max()),
            float((B == 4).float().mean()), float((B == 2).float().mean()),
            float((B == 3).float().mean()), float((B == 0).float().mean()),
            float(n_atoms), float(m),
            float(order_sum.max()), float(order_sum.mean()),
        ], dtype=np.float32)
        out.append({'B': B, 'A': A, 'smiles': smi, 'valid': bool(valid),
                    'model_score': float(lpb.sum()+lpa.sum()), 'features': feats})
    return out, {'flagged': k, 'n_pairs': m, 'candidates': R}


def label_candidates(row, candidates):
    """Reference-dependent targets; training/diagnosis only."""
    ref = row['smiles']
    for c in candidates:
        c['is_reference_species'] = bool(c['smiles'] == ref)
        c['is_reference_B'] = bool(torch.equal(c['B'], row['labels']) and row['missed'] == 0)
    return candidates
