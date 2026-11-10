"""Summarize the fixed-denominator complete-state and graph comparators."""

import argparse
import json
from pathlib import Path
import statistics

from evaluate_pose_external_states import reference_population, score_dump


FIELDS = ('strict', 'unified_resonance', 'smiles_species')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage_c_graph(path, refs, sdf, outdir):
    rows = read(path)['records']
    if len(rows) != len(refs) or {r['id'] for r in rows} != set(refs):
        raise ValueError('Stage C prediction names do not match the graph reference')
    edge_path = outdir/(path.stem+'_public_edges.jsonl')
    with edge_path.open('w', encoding='utf-8') as handle:
        for r in rows:
            name = r['id']
            mol = refs[name]
            truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                     4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}
            graph = {tuple(e): c for e, c in zip(r['pairs'], r['bond_labels'], strict=True) if c}
            for i, j in sorted(truth.keys() | graph.keys()):
                handle.write(json.dumps(dict(name=name, i=i, j=j, true=truth.get((i, j), 0), pred=graph.get((i, j), 0)))+'\n')
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    out = score_dump(edge_path, sdf, expected_n=len(refs))
    return {k: out[k] for k in FIELDS}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--dir', type=Path, required=True)
    p.add_argument('--stage-c', type=Path, required=True, nargs=3)
    a = p.parse_args()
    refs, excluded = reference_population(a.sdf)
    if len(refs) != 491 or excluded:
        raise ValueError('Unexpected public pose population')
    d = a.dir
    stage_c = [stage_c_graph(path, refs, a.sdf, d) for path in a.stage_c]
    native = read(d/'native_summary.json')
    if [r['run'] for r in native] != [path.name for path in a.stage_c] or any(r['n'] != len(refs) for r in native):
        raise ValueError('Native summary does not match these prediction files')
    for g, n in zip(stage_c, native, strict=True):
        g.update(constructible=n['constructible']/n['n'], native=n['identity_matches']/n['n'])
    rows = {'our_stage_c': {'mean': {k: statistics.mean(x[k] for x in stage_c) for k in stage_c[0]},
                            'sample_sd': {k: statistics.stdev(x[k] for x in stage_c) for k in stage_c[0]}, 'per_seed': stage_c}}
    ob = read(d/'openbabel_native.json')
    for label, graph, native in (
        ('OpenBabel + native H', read(d/'openbabel_native_native_graph_verdicts.json'), ob['openbabel_native_geometry_only']),
        ('OpenBabel + shared catalogue/Q', read(d/'openbabel_verdicts.json'), ob['catalogue_reference_Q']),
        ('YuelBond + shared catalogue/Q', read(d/'yuelbond_verdicts.json'), read(d/'yuelbond_native.json')['catalogue_reference_Q']),
    ):
        if graph['n'] != len(refs) or native['n'] != len(refs):
            raise ValueError(f'Denominator mismatch for {label}')
        rows[label] = {k: graph[k] for k in FIELDS} | {k: native[k]/len(refs) for k in ('constructible','native')}
    (d/'public_pose_baseline_summary.json').write_text(json.dumps(dict(n=len(refs), rows=rows), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
