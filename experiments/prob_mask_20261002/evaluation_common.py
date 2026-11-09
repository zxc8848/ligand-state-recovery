"""One versioned verdict schema and fixed reference population for all methods.

Reference atom states are used only to verify representation equivalence, never
as prediction inputs. Canonical SMILES is the legacy charge-blind proxy, not the
decoder's native (B,h,q) endpoint.
"""
import collections
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / 'archive/legacy_two_stage_20261003'
SCHEMA = 'bondnet.graph-verdicts.v3'
DEFINITION = 'terminal-oxy-v2+validated-aromatic-components-v3'
ALLOWED = [6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83]


def aromatic_component_align(predicted, truth, mol):
    """Accept complete, valid aromatic/Kekule forms in genuinely aromatic systems.

    Fixed reference q/H prevents sanitization from inventing hydrogens to rescue
    an invalid assignment. Fused systems are connected aromatic-bond components,
    not SSSR rings with a prescribed double-bond count.
    """
    from rdkit import Chem
    reference = Chem.Mol(mol)
    Chem.SanitizeMol(reference)
    remaining = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                 for b in reference.GetBonds() if b.GetIsAromatic()}
    aligned = dict(predicted)
    while remaining:
        component = {remaining.pop()}
        atoms = set(next(iter(component)))
        growing = True
        while growing:
            growing = False
            for edge in list(remaining):
                if atoms.intersection(edge):
                    component.add(edge); atoms.update(edge); remaining.remove(edge)
                    growing = True
        labels = [predicted.get(e, 0) for e in component]
        if not (all(c == 4 for c in labels) or all(c in (1, 2) for c in labels)):
            continue
        if any(truth.get(e, 0) not in (1, 2, 4) for e in component):
            continue
        test = Chem.Mol(reference)
        counts = [a.GetTotalNumHs(includeNeighbors=True) for a in test.GetAtoms()]
        try:
            Chem.Kekulize(test, clearAromaticFlags=True)
            for atom, h in zip(test.GetAtoms(), counts):
                atom.SetNumExplicitHs(int(h)); atom.SetNoImplicit(True)
            for i, j in component:
                c = predicted[(i, j)]
                bond = test.GetBondBetweenAtoms(i, j)
                bond.SetBondType({1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                                  4: Chem.BondType.AROMATIC}[c])
                bond.SetIsAromatic(c == 4)
                if c == 4:
                    test.GetAtomWithIdx(i).SetIsAromatic(True)
                    test.GetAtomWithIdx(j).SetIsAromatic(True)
            Chem.SanitizeMol(test)
        except Exception:
            continue
        if all(test.GetBondBetweenAtoms(i, j).GetIsAromatic() for i, j in component):
            for edge in component:
                aligned[edge] = truth[edge]
    return aligned


def reference_population(sdf, allowed=ALLOWED):
    from rdkit import Chem
    refs, excluded, all_names = {}, [], set()
    for idx, mol in enumerate(Chem.SDMolSupplier(str(sdf), removeHs=False)):
        if mol is None:
            raise ValueError(f'Unparseable reference record {idx}; denominator shrink forbidden')
        if mol.GetNumConformers() == 0 or not mol.HasProp('_Name'):
            raise ValueError(f'Reference {idx} lacks coordinates or name')
        mol = Chem.RemoveHs(mol, sanitize=True)
        name = mol.GetProp('_Name')
        if name in all_names:
            raise ValueError(f'Duplicate reference name {name!r}')
        all_names.add(name)
        if not {a.GetAtomicNum() for a in mol.GetAtoms()} <= set(allowed):
            excluded.append(name); continue
        refs[name] = mol
    if not refs:
        raise ValueError('Empty reference population')
    return refs, excluded


def load_verdicts(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    if d.get('schema') != SCHEMA or d.get('definition') != DEFINITION:
        raise ValueError(f'{path}: stale verdict schema; rescore predictions')
    rows = d['per_molecule']
    if len({r['name'] for r in rows}) != len(rows) or len(rows) != d['n']:
        raise ValueError(f'{path}: duplicate names or inconsistent n')
    return d, {r['name']: r for r in rows}


def score_dump(per_edge, sdf, allowed=ALLOWED, expected_n=None, population_manifest=None):
    sys.path.insert(0, str(LEGACY))
    from scripts.benchmark_templatefree_motifs import resonance_align_terminal_oxy
    from scripts.classify_candidate_ambiguity import classify
    from scripts.evaluate_species_recovery import _graph_candidate
    from scripts.evaluate_candidate_sets import species_signature
    refs, excluded = reference_population(sdf, allowed)
    if population_manifest:
        from population_manifest import validate_population
        checked_refs, checked_excluded = validate_population(sdf, population_manifest)
        if set(refs) != set(checked_refs) or set(excluded) != set(checked_excluded):
            raise ValueError('Scorer population differs from the frozen population')
    if expected_n is not None and len(refs) != expected_n:
        raise ValueError(f'Expected {expected_n} references, found {len(refs)}')
    per = collections.defaultdict(list)
    for line in Path(per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line); per[r['name']].append(r)
    extra = set(per) - set(refs) - set(excluded)
    if extra:
        raise ValueError(f'Unexpected prediction names: {sorted(extra)[:8]}')
    records = []
    for name, mol in refs.items():
        expected = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()}
        missing = name not in per
        truth, pred, seen = {}, {}, set()
        if missing:
            truth = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                     4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}
        else:
            for r in per[name]:
                e = tuple(sorted((int(r['i']), int(r['j']))))
                if e in seen or e[0] == e[1] or min(e) < 0 or max(e) >= mol.GetNumAtoms():
                    raise ValueError(f'{name}: duplicate or invalid edge {e}')
                seen.add(e)
                t, p = int(r['true']), int(r['pred'])
                if t not in range(5) or p not in range(6):
                    raise ValueError(f'{name}: unsupported true/pred labels {t}/{p}')
                if t: truth[e] = t
                if p: pred[e] = p
            if set(truth) != expected:
                raise ValueError(f'{name}: dump reference edge set differs from SDF')
            sdf_graph = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
                         4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())
                         for b in mol.GetBonds()}
            checked = resonance_align_terminal_oxy(truth, sdf_graph, mol, elements=(6, 7, 15, 16))
            if aromatic_component_align(checked, sdf_graph, mol) != sdf_graph:
                raise ValueError(f'{name}: dump true labels are not chemically equivalent to SDF')
        adjacency = collections.defaultdict(set)
        for i, j in truth:
            adjacency[i].add(j); adjacency[j].add(i)
        strict = not missing and pred == truth
        aligned = resonance_align_terminal_oxy(pred, truth, mol, elements=(6, 7, 15, 16))
        aligned = aromatic_component_align(aligned, truth, mol)
        resonance = not missing and aligned == truth
        verdict, _ = classify(mol, truth, pred, adjacency)
        elems = [a.GetAtomicNum() for a in mol.GetAtoms()]
        rk, _, _ = species_signature(mol, _graph_candidate(elems, truth), adjacency, return_status=True)
        pk, _, _ = species_signature(mol, _graph_candidate(elems, pred), adjacency, return_status=True)
        smiles = (not missing and not rk.startswith('fallback::')
                  and not pk.startswith('fallback::') and rk == pk)
        records.append({'name': name, 'strict': bool(strict), 'unified_resonance': bool(resonance),
                        'classify_resonance': not missing and verdict in ('identical', 'representation_equivalent'),
                        'smiles_species': bool(smiles), 'missing_prediction': missing,
                        'true_edge_hits': sum(pred.get(e, 0) == c for e, c in truth.items()),
                        'true_edge_count': len(truth),
                        'truth_hash': hashlib.sha256(json.dumps(sorted(truth.items())).encode()).hexdigest()})
    out = {'schema': SCHEMA, 'definition': DEFINITION, 'per_edge': str(per_edge),
           'reference_sdf': str(sdf), 'n': len(records), 'raw_n': len(refs)+len(excluded),
           'population_manifest': str(population_manifest) if population_manifest else None,
           'excluded_by_element': excluded,
           'missing_predictions': [r['name'] for r in records if r['missing_prediction']],
           'per_molecule': records}
    for field in ('strict', 'unified_resonance', 'classify_resonance', 'smiles_species'):
        out[field] = sum(r[field] for r in records)/len(records)
    return out
