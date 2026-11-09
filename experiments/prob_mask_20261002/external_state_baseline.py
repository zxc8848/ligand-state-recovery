"""Give an external bond graph our own state solver, and score native identity.

The native endpoint has no external head-to-head baseline because the comparators
emit a bond graph and no per-atom formal charge or hydrogen count. That is a real
gap: the paper's central extension is predicting (B,h,q), and the one endpoint
that tests h and q is reported as an absolute number only.

This closes the gap the cheapest honest way. We take an external system's bond
graph, complete it with OUR training catalogue, OUR total-charge constraint and
OUR solver, build the molecule with OUR builder, and score native identity. The
same pipeline is run on our own bond graph with the neural atom logits zeroed,
which is the matched arm: both sides then differ only in where the bonds came
from, not in how (q,h) was chosen.

Fairness notes, both stated in the output:
  * The external tool never receives Q. We nevertheless hand its graph the
    reference Q when solving for (q,h), which is the condition most favourable
    to it. Scoring it with Q=0 instead is reported alongside.
  * Candidate pairs are taken from the external dump itself (reference edges
    union its own non-zero predictions), so no prediction of its is dropped for
    falling outside our candidate radius.

No neural network runs: catalogue_logits reads only z, the candidate pairs and
the bond labels, so this is CPU-only and needs no checkpoint.

    python external_state_baseline.py --reference_sdf <sdf> \
        --external_per_edge <jsonl> --ours_per_edge <jsonl> \
        --sources joint_v1/sources.pt [--population_manifest <json>] \
        [--output <json>]
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def read_dump(path, names):
    per = collections.defaultdict(dict)
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['name'] not in names:
            continue
        per[r['name']][tuple(sorted((int(r['i']), int(r['j']))))] = (
            int(r['pred']), int(r.get('true', 0)))
    return per


def solve_and_build(J, torch, mol_row, pairs, bonds, catalogue, Q, rng):
    """Complete (q,h) for a fixed bond graph using the catalogue prior alone."""
    import numpy as np
    n = len(mol_row['z'])
    b = {'z': torch.tensor(mol_row['z']), 'pairs': torch.tensor(pairs, dtype=torch.long),
         'feat': torch.zeros(1)}
    bd = torch.tensor(bonds, dtype=torch.long)
    logits = J.catalogue_logits(b, bd, catalogue)          # neural term is absent
    state = torch.full((n,), J.AMASK, dtype=torch.long)
    ad = J.constrained_atoms(logits, state, [n], torch.tensor([Q]), rng, greedy=True)
    row = {'z': torch.tensor(mol_row['z']), 'pairs': torch.tensor(pairs, dtype=torch.long)}
    smi, good = J.build_species(row, bd, ad.reshape(-1))
    return smi, good


def run(J, torch, refs, per, catalogue, rng, force_neutral=False, adapt=False):
    from rdkit import Chem
    from kekule_adapter import kekulize_graph
    ok = built = missing = 0
    adapted_mols = 0
    for name, mol in refs.items():
        if name not in per:
            missing += 1          # scored as a failure, not dropped
            continue
        graph = {e: per[name][e][0] for e in per[name]}
        if adapt:
            # identical, reference-free normalisation for every arm: resolve
            # all-aromatic components to a Kekule form before building
            graph, converted, _ = kekulize_graph(graph)
            adapted_mols += bool(converted)
        edges = sorted(graph)
        bonds = [graph[e] for e in edges]
        bad = sorted({c for c in bonds if c < 0 or c > 4})
        if bad:
            # build_species maps 0-4 onto [none, single, double, triple, aromatic];
            # class 5 (delocalized) is an output convention applied after building,
            # so a dump carrying it cannot go through this path.
            raise SystemExit(f'{name}: bond labels {bad} outside 0-4. Use the '
                             'localmap dump, not the delocalized-rewritten one.')
        z = [a.GetAtomicNum() for a in mol.GetAtoms()]
        Q = 0 if force_neutral else sum(a.GetFormalCharge() for a in mol.GetAtoms())
        ref_smi = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=False)
        smi, good = solve_and_build(J, torch, {'z': z}, edges, bonds, catalogue, Q, rng)
        built += bool(good)
        ok += bool(good and smi == ref_smi)
    return ok, built, missing, adapted_mols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--external_per_edge', required=True)
    ap.add_argument('--ours_per_edge', required=True)
    ap.add_argument('--sources', default=str(HERE/'joint_v1/sources.pt'))
    ap.add_argument('--population_manifest', default='')
    ap.add_argument('--output', default='')
    a = ap.parse_args()

    import numpy as np
    import torch
    import joint as J
    from evaluation_common import reference_population, ALLOWED
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    refs, _ = reference_population(a.reference_sdf, ALLOWED)
    if a.population_manifest:
        from population_manifest import validate_population
        checked, _ = validate_population(a.reference_sdf, a.population_manifest)
        if set(refs) != set(checked):
            raise SystemExit('population differs from the frozen manifest')
    names = set(refs)

    # ['train'] is not optional: the catalogue is a training-split statistic, and
    # pose_localmap.py / pose_ablate_search.py build it the same way. Taking the
    # whole file would fold validation and test rows into the prior.
    loaded = torch.load(a.sources, map_location='cpu', weights_only=False)
    if not isinstance(loaded, dict) or 'train' not in loaded:
        raise SystemExit(f'{a.sources}: expected a dict with a "train" split')
    catalogue = J.make_catalogue(loaded['train'])
    print(f'catalogue from {len(loaded["train"])} training rows '
          f'(splits present: {sorted(loaded)})')
    rng = np.random.default_rng(11)

    ext = read_dump(a.external_per_edge, names)
    ours = read_dump(a.ours_per_edge, names)
    n = len(refs)
    print(f'population {n}; external covers {len(ext)}, ours covers {len(ours)}\n')

    # Ceiling for this path: the REFERENCE bond graph, completed by the same
    # catalogue-only solver and built by the same builder. Without it a low score
    # for any arm cannot be separated into 'wrong bonds' and 'this path loses
    # molecules even when the bonds are right'.
    ceiling = {name: {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                      (4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()), 0)
                      for b in mol.GetBonds()}
               for name, mol in refs.items()}

    arms = (('reference bonds (control, not a bound)', ceiling, False),
            ('external bonds, reference Q supplied',    ext, False),
            ('external bonds, Q = 0',                   ext, True),
            ('our bonds, catalogue-only states',        ours, False))
    rows = []
    for adapt in (False, True):
        head = ('as published (no representation adapter)' if not adapt
                else 'with the shared Kekule adapter applied to every arm')
        print(f'\n  --- {head}')
        for label, per, neutral in arms:
            ok, built, missing, nad = run(J, torch, refs, per, catalogue, rng,
                                          neutral, adapt)
            rows.append((label, adapt, ok, built, missing, nad))
            extra = f'   (no prediction for {missing})' if missing else ''
            tag = f'   adapter changed {nad} molecules' if adapt else ''
            print(f'    {label:40s} native {ok/n:.4f}   '
                  f'buildable {built/n:.4f}{extra}{tag}')

    ceil_rate = rows[0][2]/n
    print(f'\nAll arms share the catalogue, the sum(q)=Q solver and the builder, so')
    print(f'differences are attributable to the bond graph alone.')
    print(f'The external tool does not receive Q; the reference-Q row is the')
    print(f'condition most favourable to it.')
    print(f'\nRelative to the reference-bond control ({ceil_rate:.4f}), unadapted:')
    for l, ad, k, b, _, _ in rows:
        if ad or l.startswith('reference'):
            continue
        print(f'    {l:40s} {100*(k/n)/ceil_rate:5.1f}% of the control')
    print('\nThe control is a reference point, NOT an upper bound: it mixes builder')
    print('rejection with catalogue-only state error, and nothing here shows that')
    print('some other valid bond representation could not score higher.')
    print('\nCompare the two blocks above. If the adapter closes much of the gap,')
    print('the unadapted difference was partly representation compatibility rather')
    print('than bond-graph quality.')
    out = {'reference_sdf': str(a.reference_sdf), 'n': n,
           'arms': [{'label': l, 'adapter': ad, 'native': k, 'native_rate': k/n,
                     'buildable_rate': b/n, 'missing_predictions': m,
                     'molecules_changed_by_adapter': nad}
                    for l, ad, k, b, m, nad in rows]}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
