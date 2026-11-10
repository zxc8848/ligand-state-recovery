"""Complete fixed external graphs and score native identity over all public poses."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
from rdkit import Chem, RDLogger
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'experiments/prob_mask_20261002'))
import joint as J
from evaluation_common import reference_population, score_dump


def load_catalogue(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['format'] != 'bondnet-local-atom-state-counts-v1' or data['state_order'] != '5*(formal_charge+2)+hydrogen_count':
        raise ValueError('Unexpected catalogue format')
    result = []
    for mode in ('exact', 'coarse', 'by_element'):
        table = {}
        for key, counts in data['tables'][mode]:
            if mode == 'exact':
                key = key[0], tuple(tuple(x) for x in key[1])
            elif mode == 'coarse':
                key = key[0], tuple(key[1])
            if key in table or len(counts) != 25 or min(counts) < 0 or not sum(counts):
                raise ValueError(f'Invalid catalogue entry in {mode}')
            table[key] = np.asarray(counts, dtype=np.float64)
        result.append(table)
    return tuple(result)


def positive_dump(path, refs):
    groups = defaultdict(dict)
    seen = defaultdict(set)
    for raw in path.read_text(encoding='utf-8').splitlines():
        record = json.loads(raw)
        name = record['name']
        if name not in refs:
            raise ValueError(f'Unexpected molecule {name}')
        pair = tuple(sorted((record['i'], record['j'])))
        if not 0 <= pair[0] < pair[1] < refs[name].GetNumAtoms() or pair in seen[name]:
            raise ValueError(f'Invalid or duplicate edge {name} {pair}')
        seen[name].add(pair)
        graph = groups[name]
        if record['pred']:
            graph[pair] = record['pred']
    return groups


def build(z, graph, h, q):
    if len(z) != len(h) or len(z) != len(q):
        raise ValueError('Atom state count differs from geometry')
    if not graph or any(c not in (1, 2, 3, 4) for c in graph.values()):
        return None, False, 'empty_or_unsupported_graph'
    mol = Chem.RWMol()
    for atom_z, n_h, charge in zip(z, h, q, strict=True):
        atom = Chem.Atom(atom_z)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(n_h)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    types = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    for (i, j), label in graph.items():
        mol.AddBond(i, j, types[label])
        if label == 4:
            mol.GetAtomWithIdx(i).SetIsAromatic(True)
            mol.GetAtomWithIdx(j).SetIsAromatic(True)
    try:
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        if len(Chem.GetMolFrags(m)) != 1:
            return None, False, 'disconnected'
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            return None, False, 'radical'
        return Chem.MolToSmiles(m, isomericSmiles=False), True, None
    except (ValueError, RuntimeError):
        return None, False, 'sanitize_failure'


def catalogue_atoms(z, graph, total_charge, table, rng):
    pairs = sorted(graph)
    batch = dict(z=torch.tensor(z), pairs=torch.tensor(pairs, dtype=torch.long).reshape(-1, 2), feat=torch.zeros(1))
    labels = torch.tensor([graph[e] for e in pairs], dtype=torch.long)
    logits = J.catalogue_logits(batch, labels, table)
    atom = J.constrained_atoms(logits, torch.full((len(z),), J.AMASK, dtype=torch.long),
                              [len(z)], torch.tensor([total_charge]), rng, greedy=True).tolist()
    h, q = [s % 5 for s in atom], [s // 5 - 2 for s in atom]
    if sum(q) != total_charge:
        raise RuntimeError('Shared solver did not enforce Q')
    return h, q


def openbabel_complete(z, coords):
    from openbabel import openbabel as ob
    mol = ob.OBMol()
    for atom_z, xyz in zip(z, coords.astype(np.float32), strict=True):
        a = mol.NewAtom()
        a.SetAtomicNum(atom_z)
        a.SetVector(*map(float, xyz))
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()

    def graph_now():
        return {tuple(sorted((b.GetBeginAtomIdx()-1, b.GetEndAtomIdx()-1))):
                4 if b.IsAromatic() else int(b.GetBondOrder()) for b in ob.OBMolBondIter(mol)
                if b.GetBeginAtomIdx() <= len(z) and b.GetEndAtomIdx() <= len(z)}

    graph = graph_now()
    # OpenBabel 3's manually constructed atoms have zero implicit H until
    # their intended hydrogens are assigned, unlike atoms from an XYZ reader.
    # Use the library's standard typical-valence helper before materializing H.
    for a in ob.OBMolAtomIter(mol):
        ob.OBAtomAssignTypicalImplicitHydrogens(a)
    if not mol.AddHydrogens(False, False):
        raise ValueError('OpenBabel AddHydrogens returned failure')
    heavy = [mol.GetAtom(i+1) for i in range(len(z))]
    if [a.GetAtomicNum() for a in heavy] != z:
        raise RuntimeError('OpenBabel changed heavy atom inventory during H addition')
    h = [int(a.ExplicitHydrogenCount() + a.GetImplicitHCount()) for a in heavy]
    q = [int(a.GetFormalCharge()) for a in heavy]
    return graph, graph_now(), h, q


def run(refs, graphs, table, run_native, *, native_edges=None):
    rows = []
    rng = np.random.default_rng(11)
    for name, mol in refs.items():
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        target_q = Chem.GetFormalCharge(mol)
        target = Chem.MolToSmiles(mol, isomericSmiles=False)
        graph = graphs.get(name)
        state_error = None
        if run_native:
            try:
                produced, emitted, h, q = openbabel_complete(z, mol.GetConformer().GetPositions())
            except (ValueError, RuntimeError) as exc:
                state_error = f'{type(exc).__name__}: {exc}'
            if state_error is None and (graph is None or produced != graph):
                raise ValueError(f'{name}: native and previously scored OpenBabel bond maps differ')
            if state_error is None:
                graph = emitted
                native_edges[name] = emitted
        elif graph and set(graph.values()) <= {1, 2, 3, 4}:
            h, q = catalogue_atoms(z, graph, target_q, table, rng)
        if graph is None or not graph or set(graph.values()) - {1, 2, 3, 4} or state_error:
            rows.append(dict(name=name, constructible=False, native=False, total_charge_correct=False,
                             ref_Q=target_q, error=state_error or 'missing_or_unsupported_graph'))
            continue
        smi, good, error = build(z, graph, h, q)
        rows.append(dict(name=name, constructible=good, native=bool(good and smi == target),
                         total_charge_correct=sum(q) == target_q, ref_Q=target_q,
                         pred_Q=sum(q), smiles=smi, h=h, q=q, error=error))
    return dict(n=len(rows), constructible=sum(r['constructible'] for r in rows), native=sum(r['native'] for r in rows),
                total_charge_correct=sum(r['total_charge_correct'] for r in rows),
                failure_reasons=dict(Counter(r['error'] for r in rows if r['error'])), records=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--edges', type=Path, required=True)
    p.add_argument('--catalogue', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--method', choices=['OpenBabel', 'YuelBond', 'Knodle'], required=True)
    p.add_argument('--expected-n', type=int, default=491)
    a = p.parse_args()
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(1)
    refs, skipped = reference_population(a.sdf)
    if skipped or len(refs) != a.expected_n:
        raise ValueError('Reference population does not match the declared denominator')
    graphs = positive_dump(a.edges, refs)
    table = load_catalogue(a.catalogue)
    out = {'method': a.method, 'source_sdf': a.sdf.name, 'catalogue_reference_Q': run(refs, graphs, table, False)}
    if a.method == 'OpenBabel':
        emitted = {}
        out['openbabel_native_geometry_only'] = run(refs, graphs, table, True, native_edges=emitted)
        edges_out = a.output.with_name(a.output.stem+'_native_edges.jsonl')
        with edges_out.open('w', encoding='utf-8') as f:
            for name, graph in emitted.items():
                if any(c not in (1, 2, 3, 4) for c in graph.values()):
                    continue
                truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                         4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in refs[name].GetBonds()}
                for i, j in sorted(truth.keys() | graph.keys()):
                    f.write(json.dumps(dict(name=name, i=i, j=j, true=truth.get((i, j), 0), pred=graph.get((i, j), 0)))+'\n')
        scored = score_dump(edges_out, a.sdf, expected_n=a.expected_n)
        graph_metrics = {k: scored[k] for k in ('strict', 'unified_resonance', 'smiles_species')}
        out['openbabel_native_geometry_only']['graph_endpoints'] = graph_metrics
        out['openbabel_native_geometry_only']['n_heavy_graph_representation_changed'] = sum(emitted[n] != graphs[n] for n in emitted)
        a.output.with_name(a.output.stem+'_native_graph_verdicts.json').write_text(json.dumps(scored, indent=2)+'\n', encoding='utf-8')
    with a.output.open('w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, allow_nan=False)
        f.write('\n')
    for key, arm in out.items():
        if isinstance(arm, dict):
            print(key, {k: v for k, v in arm.items() if k != 'records'})


if __name__ == '__main__':
    main()
