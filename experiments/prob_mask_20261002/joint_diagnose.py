"""Q-embedding counterfactual and same-B multiple-species demonstrations.

Demonstration budget=32 is NOT used in eight-candidate performance tables.
No thermodynamic or calibrated-posterior claim is made.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from rdkit import RDLogger

import joint as j


@torch.no_grad()
def main():
    torch.set_num_threads(4); RDLogger.DisableLog('rdApp.*')
    data = torch.load(j.HERE/'features.pt', weights_only=False)['rows']
    ck = torch.load(j.HERE/'best.pt', weights_only=False)
    net = j.JointNet(ck['args']['width'], ck['args']['layers']).cuda().eval()
    net.load_state_dict(ck['state'], strict=True)
    catalogue = j.make_catalogue(torch.load(j.HERE/'sources.pt', weights_only=False)['train'])
    probes = []
    for sigma in [0., .2]:
        rows = [r for r in data['val'] if r['sigma'] == sigma and r['Q'] != 0]
        probability_delta = total_nodes = changed_atoms = changed_graphs = correct = no_embed_correct = 0
        true_CE = zero_CE = 0
        for start in range(0, len(rows), 32):
            rr = rows[start:start+32]; b = j.pack(rr, 'cuda')
            bs = torch.full_like(b['labels'], j.BMASK); ats = torch.full_like(b['atarget'], j.AMASK)
            actual = net(b, bs, ats)
            counterfactual = net(dict(b, Q=torch.zeros_like(b['Q'])), bs, ats)
            probability_delta += (actual[1].softmax(-1)-counterfactual[1].softmax(-1)).abs().sum().item()
            total_nodes += len(ats)
            true_CE += nn.functional.cross_entropy(actual[1], b['atarget'], reduction='sum').item()
            zero_CE += nn.functional.cross_entropy(counterfactual[1], b['atarget'], reduction='sum').item()
            # Both decoders STILL enforce the true supplied Q. Only neural Q
            # embedding is ablated, not the hard charge condition in the DP.
            bd, ad = j.sample(net, b, 1, 22, greedy=True, cached_logits=actual)
            b0, a0 = j.sample(net, b, 1, 22, greedy=True, cached_logits=counterfactual)
            for r, x, y, xx, yy in zip(rr, bd.cpu().split(b['lengths']), ad.cpu().split(b['atom_lengths']),
                                      b0.cpu().split(b['lengths']), a0.cpu().split(b['atom_lengths'])):
                changed_atoms += not torch.equal(y, yy); changed_graphs += not torch.equal(x, xx)
                correct += j.build_species(r, x, y)[0] == r['smiles']
                no_embed_correct += j.build_species(r, xx, yy)[0] == r['smiles']
        probes.append({'sigma': sigma, 'n_net_charged': len(rows),
                       'mean_absolute_atom_probability_change': probability_delta/(total_nodes*25),
                       'joint_atom_CE_reference_Q_embedding': true_CE/total_nodes,
                       'joint_atom_CE_zero_Q_embedding': zero_CE/total_nodes,
                       'molecules_with_changed_atom_assignment_after_same_Q_DP': int(changed_atoms),
                       'molecules_with_changed_B': int(changed_graphs),
                       'greedy_species_reference_embedding': correct/len(rows),
                       'greedy_species_zero_embedding_same_Q_DP': no_embed_correct/len(rows)})
    rows = [r for r in data['val'] if r['sigma'] == 0 and r['domain'] == 'diverse_charge'][:64]
    demos = []
    for start in range(0, len(rows), 16):
        rr = rows[start:start+16]; b = j.pack(rr, 'cuda')
        cached = net(b, torch.full_like(b['labels'], j.BMASK), torch.full_like(b['atarget'], j.AMASK))
        groups = [{} for _ in rr]
        for k in range(32):
            bd, ad = j.sample(net, b, 1, 20261002+start*97+k*1000000, cached_logits=cached,
                              atoms_after_B=True, catalogue=catalogue, combine_catalogue=True)
            for g, (r, x, y) in enumerate(zip(rr, bd.cpu().split(b['lengths']), ad.cpu().split(b['atom_lengths']))):
                smi, good = j.build_species(r, x, y)
                if good:
                    groups[g].setdefault(tuple(x.tolist()), {}).setdefault(smi, y.tolist())
        for r, group in zip(rr, groups):
            for B, species in group.items():
                if len(species) > 1:
                    demos.append({'id': r['id'], 'domain': r['domain'], 'Q': r['Q'],
                                  'note': 'Identical heavy Z,R and identical sampled typed B; multiple closed-shell species with different h/q. Energetic plausibility not checked.',
                                  'Z': r['z'].tolist(), 'R': r['coord'].tolist(),
                                  'sampled_B': [(ij, c) for ij, c in zip(r['pairs'].tolist(), B) if c],
                                  'species': [{'smiles': smi, 'h': [s % 5 for s in state], 'q': [s//5-2 for s in state]} for smi, state in species.items()]})
                    break
    result = {'Q_embedding_diagnostics': probes, 'same_B_demonstration_count': len(demos),
              'demonstration_scanned': len(rows), 'demonstration_candidates': 32,
              'scope': 'Developmental diagnostics. The 32-candidate demo is separate from the raw budget=8 main experiment.',
              'same_B_examples': demos[:10]}
    (j.HERE/'diagnostics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    for r in probes:
        print(r)
    print('same-B distinct-species demos:', len(demos), '/', len(rows), 'at separate budget=32')


if __name__ == '__main__':
    main()
