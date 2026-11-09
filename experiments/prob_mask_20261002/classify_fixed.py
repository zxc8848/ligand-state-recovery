"""The Kekule-equivalence hole in classify(), fixed, without touching the archive.

`archive/.../classify_candidate_ambiguity.py:155` accepts a predicted {1,2}
labelling of a complete aromatic system as a representation variant whenever

    pred_labels.count(2) == len(atoms) // 2

i.e. it checks only the NUMBER of double bonds, never whether they alternate.
A benzene ring predicted with three ADJACENT double bonds passes that test and is
declared `representation_equivalent`, although it is not a Kekule structure of
benzene at all.

The fix is a necessary condition for a Kekule structure: no atom of the system may
carry two or more double bonds inside the system, and the count must still match.
Pyrrole-type systems are unaffected -- an atom with no incident double (the
lone-pair donor) is allowed; only atoms with two or more are rejected.

This module does NOT edit the archived scorer. Editing it would silently change
every published number. Instead it re-checks each molecule and reports how many
verdicts the fix flips, so the archived numbers can either stand or be corrected
with the difference stated.

Usage:
    python classify_fixed.py --per_edge <dump.jsonl> --reference_sdf <ref.sdf>
    python classify_fixed.py --selftest
"""
import argparse
import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT/'archive/legacy_two_stage_20261003'


def kekule_alternates(system, pred_labels):
    """Necessary condition for a Kekule structure over one aromatic system.

    system      iterable of (i, j) edges forming a complete aromatic system
    pred_labels the predicted label per edge, in the same order

    Returns True when the doubles could form a Kekule alternation: the count is
    len(atoms)//2 and no atom carries two or more doubles inside the system.
    """
    edges = list(system)
    atoms = {a for e in edges for a in e}
    if len(edges) != len(pred_labels):
        raise ValueError('system and pred_labels must align')
    if sum(1 for x in pred_labels if x == 2) != len(atoms) // 2:
        return False
    deg = collections.Counter()
    for (i, j), lab in zip(edges, pred_labels):
        if lab == 2:
            deg[i] += 1
            deg[j] += 1
    return all(v <= 1 for v in deg.values())


def _selftest():
    benzene = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    assert kekule_alternates(benzene, [2, 1, 2, 1, 2, 1]), 'proper Kekule rejected'
    assert kekule_alternates(benzene, [1, 2, 1, 2, 1, 2]), 'the other Kekule rejected'
    # the counterexample: three ADJACENT doubles. The archived rule accepts it
    # because the count is right; this one rejects it because atoms 1 and 2 each
    # carry two doubles.
    adjacent = [2, 2, 2, 1, 1, 1]
    assert sum(1 for x in adjacent if x == 2) == len({a for e in benzene for a in e}) // 2
    assert not kekule_alternates(benzene, adjacent), 'adjacent doubles accepted'
    # pyrrole-like: 5 atoms, 2 doubles, the donor carries none
    pyrrole = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    assert kekule_alternates(pyrrole, [2, 1, 2, 1, 1]), 'pyrrole Kekule rejected'
    assert not kekule_alternates(pyrrole, [2, 2, 1, 1, 1]), 'pyrrole adjacent accepted'
    print('classify_fixed._selftest: count-only rule accepts adjacent doubles, '
          'the position check rejects them  OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_edge')
    ap.add_argument('--reference_sdf')
    ap.add_argument('--allowed_atomic_nums', type=int, nargs='+',
                    default=[6, 7, 8, 9, 14, 15, 16, 17, 33, 35, 53, 83])
    ap.add_argument('--output', default='')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        _selftest()
        return
    if not (a.per_edge and a.reference_sdf):
        raise SystemExit('need --per_edge and --reference_sdf (or --selftest)')

    sys.path.insert(0, str(LEGACY))
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    from scripts.classify_candidate_ambiguity import classify, aromatic_systems

    per_edge = collections.defaultdict(list)
    for line in Path(a.per_edge).read_text(encoding='utf-8').splitlines():
        if line.strip():
            row = json.loads(line)
            per_edge[row['name']].append(row)
    allowed = set(a.allowed_atomic_nums)

    total = flipped = equiv = 0
    flips = []
    for mol in Chem.SDMolSupplier(a.reference_sdf, removeHs=False):
        if mol is None or mol.GetNumConformers() == 0:
            continue
        mol = Chem.RemoveHs(mol, sanitize=True)
        if not {at.GetAtomicNum() for at in mol.GetAtoms()} <= allowed:
            continue
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else None
        if name not in per_edge:
            continue
        total += 1
        truth, predicted = {}, {}
        for row in per_edge[name]:
            e = (row['i'], row['j'])
            if row['true'] > 0:
                truth[e] = row['true']
            if row['pred'] > 0:
                predicted[e] = row['pred']
        adjacency = collections.defaultdict(set)
        for i, j in truth:
            adjacency[i].add(j)
            adjacency[j].add(i)
        verdict, _ = classify(mol, truth, predicted, adjacency)
        was_equiv = verdict in ('identical', 'representation_equivalent')
        equiv += was_equiv
        if not was_equiv:
            continue
        # re-check every complete aromatic system the archived rule may have
        # waved through on the count alone
        bad = False
        for system in aromatic_systems(mol):
            system = sorted(system)
            labs = [predicted.get(e, predicted.get((e[1], e[0]), 0)) for e in system]
            if set(labs) <= {1, 2} and not kekule_alternates(system, labs):
                bad = True
                break
        if bad:
            flipped += 1
            flips.append(name)
    print(f'molecules scored                        {total}')
    print(f'archived classify called equivalent     {equiv}')
    print(f'of those, rejected by the position check {flipped}'
          + (f'   ({", ".join(flips[:8])}{"..." if len(flips) > 8 else ""})' if flips else ''))
    print(f'corrected resonance-aware recovery      {(equiv-flipped)/total:.4f}'
          f'   (archived {equiv/total:.4f})')
    if a.output:
        Path(a.output).write_text(json.dumps(
            {'per_edge': a.per_edge, 'n': total, 'archived_equivalent': equiv,
             'rejected_by_position_check': flipped, 'names': flips}, indent=1),
            encoding='utf-8')
        print('wrote', a.output)


if __name__ == '__main__':
    main()
