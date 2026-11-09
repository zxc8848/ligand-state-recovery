"""Does a second pass help? Iterative masked refinement on top of the
B -> (h,q) + catalogue + charge-DP configuration, evaluated on the 488 PDBBind
poses.

Three decoding schemes, identical network and catalogue, identical inputs:

  oneshot    the current baseline: all-MASK B in one pass, then (h,q) conditioned
             on the complete sampled B, catalogue prior, exact charge-sum DP.
  confidence MaskGIT-style: after the one-shot pass, re-mask the lowest-confidence
             fraction of edges and re-predict them conditioned on the rest; repeat.
  violation  only re-mask edges incident to atoms that RDKit itself reports as
             chemically impossible (AtomValenceException / KekulizeException) for
             the current (B,h,q); repeat. Molecules that already build are left
             untouched, so this is a repair pass, not a resampling pass.

The existing multi-round experiment (``mask_joint`` rounds 1/2/4) was run in the
independent-joint family, i.e. without the catalogue prior and without the
B-then-(h,q) factorization, so it does not tell us whether refinement helps the
configuration that is actually deployed. That is what this script measures.
"""
import argparse
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
    a.add_argument('--checkpoint', default=str(ROOT/'checkpoints/revision_v3_one_stage_heavy_seed42/best_e2e.pt'))
    a.add_argument('--joint_ckpt', default=str(HERE/'joint_v1/best.pt'))
    a.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    a.add_argument('--outdir', default=str(HERE/'joint_pose'))
    a.add_argument('--device', default='cpu')
    a.add_argument('--batch', type=int, default=16)
    a.add_argument('--sigma', type=float, default=0.0)
    a.add_argument('--rounds', type=int, default=3, help='extra refinement passes')
    a.add_argument('--fraction', type=float, default=0.15,
                   help='fraction of edges re-masked per confidence round')
    a.add_argument('--seed', type=int, default=7)
    a.add_argument('--limit', type=int, default=0)
    return a.parse_args()


def build_raw(row, bond, atom_state):
    """RWMol from decoded (B,h,q) without sanitization."""
    rw = Chem.RWMol()
    for z, state in zip(row['z'].tolist(), atom_state.tolist()):
        at = Chem.Atom(int(z))
        at.SetFormalCharge(int(state)//5-2)
        at.SetNumExplicitHs(int(state) % 5)
        at.SetNoImplicit(True)
        rw.AddAtom(at)
    types = [None, Chem.BondType.SINGLE, Chem.BondType.DOUBLE,
             Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
    for (i, j), c in zip(row['pairs'].tolist(), bond.tolist()):
        if c:
            rw.AddBond(int(i), int(j), types[int(c)])
            if c == 4:
                rw.GetAtomWithIdx(int(i)).SetIsAromatic(True)
                rw.GetAtomWithIdx(int(j)).SetIsAromatic(True)
    return rw.GetMol()


def offending_atoms(row, bond, atom_state):
    """Atom indices RDKit itself calls impossible; empty when the graph builds."""
    mol = build_raw(row, bond, atom_state)
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.AtomValenceException as exc:
        try:
            return [exc.GetAtomIdx()]
        except Exception:
            return []
    except Chem.rdchem.KekulizeException as exc:
        try:
            return list(exc.GetAtomIndices())
        except Exception:
            return []
    except Exception:
        return []
    bad = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons()]
    if bad:
        return bad
    if len(Chem.GetMolFrags(mol)) != 1:
        return []
    return []


@torch.no_grad()
def decode(net, b, catalogue, scheme, rounds, fraction, seed, lengths, alengths, rows):
    """Return (bond draws, atom draws) per molecule for one decoding scheme."""
    gen = torch.Generator(device=b['feat'].device).manual_seed(seed)
    rng = np.random.default_rng(seed)
    full_mask = torch.full_like(b['labels'], J.BMASK)
    all_atom_mask = torch.full_like(b['atarget'], J.AMASK)

    def atoms_given_B(bd):
        _, al = net(b, bd, all_atom_mask)
        al = al + J.catalogue_logits(b, bd, catalogue)
        return J.constrained_atoms(al, all_atom_mask, b['atom_lengths'], b['Q'], rng, greedy=True)

    bl, _ = net(b, full_mask, all_atom_mask)
    bp = bl.softmax(-1)
    bd = bp.argmax(-1)
    conf = bp.gather(1, bd[:, None]).squeeze(1)
    ad = atoms_given_B(bd)
    history = [(bd.clone(), ad.clone())]

    if scheme == 'oneshot':
        return history

    for _ in range(rounds):
        state = bd.clone()
        remask = torch.zeros_like(bd, dtype=torch.bool)
        if scheme == 'confidence':
            start = 0
            for n in lengths:
                if n:
                    k = max(1, int(round(fraction*n)))
                    local = conf[start:start+n]
                    idx = torch.topk(-local, k).indices
                    remask[start+idx] = True
                start += n
        elif scheme == 'violation':
            bsplit = bd.cpu().split(lengths)
            asplit = ad.cpu().split(alengths)
            estart = 0
            node_offset = 0
            for g, r in enumerate(rows):
                bad = offending_atoms(r, bsplit[g], asplit[g])
                if bad:
                    pairs = r['pairs']
                    bad = set(int(x) for x in bad)
                    hit = torch.tensor([(int(i) in bad) or (int(j) in bad)
                                        for i, j in pairs.tolist()],
                                       dtype=torch.bool, device=remask.device)
                    remask[estart:estart+lengths[g]] = hit
                estart += lengths[g]
                node_offset += alengths[g]
        if not remask.any():
            history.append((bd.clone(), ad.clone()))
            continue
        state[remask] = J.BMASK
        bl2, _ = net(b, state, all_atom_mask)
        bp2 = bl2.softmax(-1)
        draw = bp2.argmax(-1)
        bd = torch.where(remask, draw, bd)
        conf = torch.where(remask, bp2.gather(1, draw[:, None]).squeeze(1), conf)
        ad = atoms_given_B(bd)
        history.append((bd.clone(), ad.clone()))
    return history


def main():
    a = parse_args()
    RDLogger.DisableLog('rdApp.*')
    p.seed_all(11)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

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
    rng = np.random.default_rng(11)
    geos = [p.geometry(r, a.sigma, rng, bargs['cutoff']) for r in rows]
    feats = []
    for i in range(0, len(geos), a.batch):
        feats.extend(J.extract(backbone, geos[i:i+a.batch], a.device))

    # load_joint picks JointNet or the t-conditioned JointNetT from the
    # checkpoint's recorded arch, so both are evaluated by this same code.
    net, ck = T.load_joint(a.joint_ckpt, a.device)
    catalogue = J.make_catalogue(torch.load(a.sources, map_location='cpu', weights_only=False)['train'])

    schemes = ['oneshot', 'confidence', 'violation']
    per_round = {s: [] for s in schemes}
    timing = {}
    for scheme in schemes:
        t0 = time.perf_counter()
        records = []
        for i in range(0, len(feats), a.batch):
            chunk = feats[i:i+a.batch]
            b = J.pack(chunk, a.device)
            hist = decode(net, b, catalogue, scheme, a.rounds, a.fraction,
                          a.seed, b['lengths'], b['atom_lengths'], chunk)
            per_pass = []
            for bd, ad in hist:
                per_pass.append((bd.cpu().split(b['lengths']), ad.cpu().split(b['atom_lengths'])))
            for g, r in enumerate(chunk):
                rec = {'name': r['id'], 'passes': []}
                for bs, ats in per_pass:
                    smi, good = J.build_species(r, bs[g], ats[g])
                    rec['passes'].append({
                        'species': bool(smi == r['smiles']),
                        'valid': bool(good),
                        'strictB': bool(torch.equal(bs[g], r['labels']) and r['missed'] == 0),
                        'joint': bool(torch.equal(bs[g], r['labels']) and r['missed'] == 0
                                      and torch.equal(ats[g], r['atom_target'])),
                    })
                records.append(rec)
            if i == 0:
                p.log(f'{scheme}: first batch has {len(per_pass)} passes')
        timing[scheme] = time.perf_counter()-t0
        n = len(records)
        npass = min(len(r['passes']) for r in records)
        for k in range(npass):
            per_round[scheme].append({
                'pass': k,
                'species': sum(r['passes'][k]['species'] for r in records)/n,
                'strictB': sum(r['passes'][k]['strictB'] for r in records)/n,
                'joint': sum(r['passes'][k]['joint'] for r in records)/n,
                'valid': sum(r['passes'][k]['valid'] for r in records)/n,
            })
        p.log(f'{scheme} ({timing[scheme]:.1f}s): ' + ' | '.join(
            f"pass{r['pass']} species={r['species']:.4f} strictB={r['strictB']:.4f} valid={r['valid']:.4f}"
            for r in per_round[scheme]))
        if scheme == schemes[-1]:
            Path(out/'pose_iterate_records.json').write_text(
                json.dumps(records, indent=1), encoding='utf-8')

    Path(out/'pose_iterate_summary.json').write_text(json.dumps(
        {'args': vars(a), 'n': len(feats), 'per_round': per_round, 'seconds': timing},
        indent=1), encoding='utf-8')
    p.log('wrote pose_iterate_summary.json')


if __name__ == '__main__':
    main()
