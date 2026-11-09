"""Local stage-C inference from heavy elements, coordinates and supplied Q.

Uses the published search recipe, without reference states or training rows.
This is not the frozen evaluation driver. Inputs/outputs are not for upload.
"""
import argparse
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
from rdkit import RDLogger

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/prob_mask_20261002'))
import joint as J
import probe as P
from bondnet.model.bondnet import BondNet


def read_catalogue(path):
    obj = json.loads(path.read_text(encoding='utf-8'))
    names = ('exact', 'coarse', 'by_element')
    if (set(obj) != {'format', 'state_order', 'tables'} or
            obj['format'] != 'bondnet-local-atom-state-counts-v1' or
            obj['state_order'] != '5*(formal_charge+2)+hydrogen_count' or
            set(obj['tables']) != set(names)):
        raise ValueError('Unsupported catalogue format or state order')

    def integer(x, lo, hi):
        return type(x) is int and lo <= x <= hi

    tables = []
    for name in names:
        table = {}
        for key, counts in obj['tables'][name]:
            if (not isinstance(counts, list) or len(counts) != 25 or
                    not all(type(v) is int and v >= 0 for v in counts) or not sum(counts)):
                raise ValueError(f'Invalid atom-state counts in {name}')
            if name == 'by_element':
                if not integer(key, 2, 118):
                    raise ValueError('Invalid element key')
            else:
                z, neighbours = key
                if not integer(z, 2, 118) or not isinstance(neighbours, list):
                    raise ValueError(f'Invalid element or neighbourhood in {name}')
                if name == 'exact':
                    if not all(len(p) == 2 and integer(p[0], 2, 118) and
                               integer(p[1], 1, 4) for p in neighbours):
                        raise ValueError('Invalid exact-neighbour key')
                    neighbours = [tuple(p) for p in neighbours]
                elif not all(integer(c, 1, 4) for c in neighbours):
                    raise ValueError('Invalid coarse-neighbour key')
                if neighbours != sorted(neighbours):
                    raise ValueError(f'Unsorted neighbourhood in {name}')
                key = (z, tuple(neighbours))
            if key in table:
                raise ValueError(f'Duplicate key in {name}')
            table[key] = np.asarray(counts, dtype=np.float64)
        if not table:
            raise ValueError(f'Empty catalogue table: {name}')
        tables.append(table)
    return tuple(tables)


def load_encoder(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    args = checkpoint['args']
    model = BondNet(num_interactions=args['num_interactions'], hidden_size=args['hidden_size'],
                    cutoff=args['cutoff'], edge_embedding_size=args['edge_embedding_size'],
                    vector_norm_limit=args.get('vector_norm_limit', 0), dropout=0,
                    backbone=args.get('backbone', 'painn')).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return model.eval(), args['cutoff']


def load_head(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if set(checkpoint) != {'state', 'args'} or set(checkpoint['args']) != {'width', 'layers'}:
        raise ValueError('Expected a minimal joint-head checkpoint')
    net = J.JointNet(**checkpoint['args']).to(device)
    net.load_state_dict(checkpoint['state'], strict=True)
    return net.eval()


def prepare_row(raw, cutoff, sigma, rng):
    if set(raw) != {'id', 'Z', 'R', 'Q'}:
        raise ValueError('Expected only id, Z (atomic numbers), R (angstrom), and Q')
    name, z, coords, charge = (raw[k] for k in ('id', 'Z', 'R', 'Q'))
    if (not isinstance(name, str) or not name or not isinstance(z, list) or len(z) < 2 or
            not all(type(i) is int and 2 <= i <= 118 for i in z)):
        raise ValueError('Expected a name and at least two heavy atoms')
    if type(charge) is not int or abs(charge) > 2 * len(z):
        raise ValueError(f'{name}: Q not an integer or infeasible')
    xyz = torch.tensor(coords, dtype=torch.float32)
    if xyz.shape != (len(z), 3) or not torch.isfinite(xyz).all():
        raise ValueError(f'{name}: invalid coordinates')
    # Geometry reads truth only to build label bookkeeping; this input has no truth.
    # Atarget/labels are used for tensor shapes only; the model receives MASK.
    row = {'id': name, 'z': torch.tensor(z), 'coord': xyz, 'Q': charge, 'truth': {},
           'atom_target': torch.full((len(z),), J.AMASK, dtype=torch.long)}
    result = P.geometry(row, sigma, rng, cutoff)
    if not len(result['pairs']):
        raise ValueError(f'{name}: no candidate edges within the encoder cutoff')
    return result


def decode_row(r, net, catalogue, device, draws, trace):
    # Stage C search settings from the frozen driver (fraction .10, k <= 8,
    # top-2 per edge, beam <= 48, marginal-greedy charge solver).
    b = J.pack([r], device)
    all_b = torch.full_like(b['labels'], J.BMASK)
    all_a = torch.full_like(b['atarget'], J.AMASK)
    logits, _ = net(b, all_b, all_a)
    log_b = logits.log_softmax(-1)
    base = log_b.argmax(-1)
    conf = log_b.gather(1, base[:, None]).squeeze(1)
    k = min(8, max(1, int(round(0.10 * len(base)))))
    flagged = torch.topk(-conf, k).indices.tolist()
    options = [torch.topk(log_b[e], 2).indices.tolist() for e in flagged]
    combos = [(sum(float(log_b[e, c]) for e, c in zip(flagged, choice)), choice)
              for choice in itertools.product(*options)]
    combos.sort(key=lambda x: -x[0])
    candidates = []
    for _, choice in combos[:48]:
        v = base.clone()
        for e, c in zip(flagged, choice):
            v[e] = c
        candidates.append(v)
    assert torch.equal(candidates[0], base)
    packed = J.pack([r] * len(candidates), device)
    states_b = torch.cat(candidates)
    masked_a = torch.full_like(packed['atarget'], J.AMASK)
    _, logits_a = net(packed, states_b, masked_a)
    logits_a = logits_a + J.catalogue_logits(packed, states_b, catalogue)
    log_a = logits_a.log_softmax(-1)
    states_a = J.constrained_atoms(logits_a, masked_a, packed['atom_lengths'], packed['Q'],
                                   draws, greedy=True)
    split_b = states_b.cpu().split(packed['lengths'])
    split_a = states_a.cpu().split(packed['atom_lengths'])
    log_a_split = log_a.cpu().split(packed['atom_lengths'])
    details = []
    for v, bv, av, al in zip(candidates, split_b, split_a, log_a_split):
        smiles, good = J.build_species(r, bv, av)
        bscore = float(log_b.gather(1, v[:, None]).sum())
        ascore = float(al.gather(1, av[:, None]).sum())
        details.append({'bond_labels': bv.tolist(), 'atom_states': av.tolist(), 'smiles': smiles,
                        'good': bool(good), 'score': bscore + ascore, 'bscore': bscore,
                        'ascore': ascore, 'dist': int((bv != base.cpu()).sum())})
    c = max(range(len(details)), key=lambda i: (details[i]['good'], details[i]['score']))
    b = max(range(len(details)), key=lambda i: (details[i]['good'], details[i]['bscore']))
    selected = details[c]
    q = [x // 5 - 2 for x in selected['atom_states']]
    if sum(q) != r['Q']:
        raise RuntimeError('Predicted formal charges did not sum to the input Q')
    result = {'id': r['id'], 'Q': r['Q'], 'pairs': r['pairs'].tolist(), 'flagged': k,
              'n_candidates': len(details), 'selected_indices': {'greedy': 0, 'bondonly': b, 'localmap': c},
              **selected, 'q': q, 'h': [x % 5 for x in selected['atom_states']]}
    if trace:
        result['candidates_detail'] = details
    return result


@torch.inference_mode()
def predict(records, encoder, head, catalogue, *, device='cpu', batch=16, sigma=0., trace=False):
    if not records or batch < 1 or not np.isfinite(sigma) or sigma < 0:
        raise ValueError('Empty inputs or invalid batch/noise setting')
    P.seed_all(11)
    RDLogger.DisableLog('rdApp.*')
    model, cutoff = load_encoder(encoder, device)
    net = load_head(head, device)
    tables = read_catalogue(catalogue)
    rng, draws = np.random.default_rng(11), np.random.default_rng(11)
    rows = [prepare_row(r, cutoff, sigma, rng) for r in records]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input identifiers')
    output = []
    for i in range(0, len(rows), batch):
        for row in J.extract(model, rows[i:i + batch], device):
            output.append(decode_row(row, net, tables, device, draws, trace))
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for flag in ('input', 'encoder', 'head', 'catalogue', 'output'):
        ap.add_argument('--' + flag, type=Path, required=True)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--sigma', type=float, default=0., help='diagnostic noise, default off')
    ap.add_argument('--trace', action='store_true', help='record candidates for local parity testing')
    a = ap.parse_args()
    if a.output.exists():
        ap.error('Output already exists; use a new file')
    rows = json.loads(a.input.read_text(encoding='utf-8'))
    if not isinstance(rows, list):
        ap.error('Input JSON must be an array of molecules')
    result = predict(rows, a.encoder, a.head, a.catalogue, device=a.device, batch=a.batch,
                     sigma=a.sigma, trace=a.trace)
    with a.output.open('x', encoding='utf-8') as f:
        json.dump({'scope': 'stage-C supplied-Q inference', 'records': result}, f, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
