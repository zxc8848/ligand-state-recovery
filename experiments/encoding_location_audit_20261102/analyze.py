"""Read-only audits of saved graphs; no inference or historical rescoring.

Class-4 ring membership is measured on each supplied graph's own positive-edge
topology. Atom types come from the observed heavy atoms. Reference bond labels
are not used to decide the predicted graph's ring/terminal-oxygen membership.
"""
import collections
import hashlib
import json
import statistics
from pathlib import Path

from rdkit import Chem, RDLogger

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EXP = ROOT / 'experiments/prob_mask_20261002'
ALLOWED = {6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_graphs(path, names):
    out = collections.defaultdict(dict)
    seen = set()
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if row['name'] not in names:
            continue
        edge = tuple(sorted((int(row['i']), int(row['j']))))
        key = (row['name'], edge)
        if key in seen:
            raise ValueError(f'Duplicate edge: {key}')
        seen.add(key)
        out[row['name']]  # retain a molecule even if all predictions are zero
        if row['pred']:
            out[row['name']][edge] = int(row['pred'])
    if set(out) != set(names):
        raise ValueError('Saved graph population differs from reference population')
    return out


def location_counts(graphs, refs):
    counts = collections.Counter()
    records = []
    for name, mol in refs.items():
        graph = graphs[name]
        degree = collections.Counter(i for edge in graph for i in edge)
        rw = Chem.RWMol()
        for atom in mol.GetAtoms():
            rw.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        for i, j in graph:
            rw.AddBond(i, j, Chem.BondType.SINGLE)
        topology = rw.GetMol()
        Chem.FastFindRings(topology)
        aromatic = [e for e, label in graph.items() if label == 4]
        nonring = [e for e in aromatic
                   if not topology.GetBondBetweenAtoms(*e).IsInRing()]
        terminal = [e for e in aromatic if any(
            mol.GetAtomWithIdx(i).GetAtomicNum() == 8 and degree[i] == 1 for i in e)]
        row = {'name': name, 'positive_edges': len(graph),
               'class4_edges': len(aromatic), 'class4_nonring_edges': len(nonring),
               'class4_terminal_oxygen_edges': len(terminal),
               'molecules_with_class4': int(bool(aromatic)),
               'molecules_with_nonring_class4': int(bool(nonring)),
               'molecules_with_terminal_oxygen_class4': int(bool(terminal))}
        counts.update({k: v for k, v in row.items() if k != 'name'})
        records.append(row)
    return {'counts': dict(counts), 'per_molecule': records}


def permutation_audit(paths):
    runs = [{r['name']: r for r in json.loads(p.read_text())['records']} for p in paths]
    if any(set(r) != set(runs[0]) for r in runs):
        raise ValueError('Permutation populations differ; intersection is not sufficient')
    checked = nontrivial = restored = 0
    for name, base in runs[0].items():
        bz = base['z']
        btruth = sorted(tuple(e) for e in base['ref_edges'])
        for run in runs[1:]:
            row = run[name]
            perm = row['permutation']
            if sorted(perm) != list(range(len(bz))):
                raise ValueError('Invalid permutation')
            nontrivial += perm != list(range(len(bz)))
            mapped_z = {old: row['z'][new] for new, old in enumerate(perm)}
            assert [mapped_z[i] for i in range(len(bz))] == bz
            mapped_truth = []
            for i, j, zi, zj, label in row['ref_edges']:
                oi, oj = perm[i], perm[j]
                if oi > oj:
                    oi, oj, zi, zj = oj, oi, zj, zi
                mapped_truth.append((oi, oj, zi, zj, label))
            assert sorted(mapped_truth) == btruth
            assert len(row['localmap_states']) == len(bz)
            checked += 1
        restored += 1
    assert nontrivial == checked
    return {'n_molecules': len(runs[0]), 'n_permutations': len(runs)-1,
            'checked_permutations': checked, 'nontrivial_permutations': nontrivial,
            'molecules_with_reference_atom_and_edge_inverse_map_verified': restored}


def main():
    RDLogger.DisableLog('rdApp.*')
    sdf = ROOT / 'data/pdbbind_pose_locked500.sdf'
    ours = EXP / 'locked_eval/seed20261002/pose_localmap_locked_fullgraph.jsonl'
    knodle = EXP / 'locked_eval/knodle_per_edge.jsonl'
    ladder = EXP / 'locked_eval/component_ladder.json'
    deloc = [EXP / f'joint_v6/seed{s}/pdbbind_sigma0p00/pose_delocalize_fullgraph.json'
             for s in (20261002, 20261003, 20261004)]
    perms = [ROOT / f'work/reviewer_checks_20261102/pose_localmap_perm{k}.json'
             for k in range(4)]
    inputs = [sdf, ours, knodle, ladder, *deloc, *perms]
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    refs = {}
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False):
        if mol is None:
            raise ValueError('Unparseable reference')
        mol = Chem.RemoveHs(mol)
        if {a.GetAtomicNum() for a in mol.GetAtoms()} <= ALLOWED:
            refs[mol.GetProp('_Name')] = mol
    assert len(refs) == 491
    reference = {name: {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                        4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())
                        for b in mol.GetBonds()} for name, mol in refs.items()}
    graphs = {'reference': reference, 'ours_seed20261002': read_graphs(ours, refs),
              'knodle': read_graphs(knodle, refs)}
    counts = {k: location_counts(g, refs) for k, g in graphs.items()}
    sel = json.loads(ladder.read_text())['selectors']
    d1, d2, full = [sel[k] for k in ('d<=1 + filter', 'd<=2 + filter', 'full localmap')]
    def delta(a, b):
        values = [100*(x-y) for x, y in zip(a, b)]
        return {'per_seed_pp': values, 'mean_pp': statistics.mean(values),
                'sample_sd_pp': statistics.stdev(values)}
    out = {'schema': 'bondnet.saved-output-location-audit.v1', 'n': len(refs),
           'scope': 'post hoc saved-output diagnostics; no new inference or scoring',
           'graph_locations': counts,
           'ladder_one_to_two': delta(d2, d1),
           'ladder_one_to_full': delta(full, d1),
           'development_delocalized_rewrites': [json.loads(p.read_text()) for p in deloc],
           'permutation_validation': permutation_audit(perms)}
    assert {str(p.relative_to(ROOT)): digest(p) for p in inputs} == before
    out['inputs_unchanged'] = True
    out['input_sha256'] = before
    out['analysis_sha256'] = digest(Path(__file__))
    (HERE / 'SUMMARY.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({k: v['counts'] for k, v in counts.items()}, indent=2))
    print(json.dumps({k: out[k] for k in ('ladder_one_to_two', 'ladder_one_to_full',
                                         'permutation_validation', 'inputs_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
