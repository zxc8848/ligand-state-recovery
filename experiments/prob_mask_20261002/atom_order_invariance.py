"""Does renumbering a molecule's atoms change what the decoder returns?

The default charge solver (joint.constrained_atoms with greedy=True) marginalises
h, runs a sum-product suffix recursion and picks q position by position. That is
order-dependent by construction, and the Methods section says so. The evidence
offered there is on random logits (9.2% suboptimal on two atoms, 28/200 order
flips on three), which bounds the solver in isolation but does not tell a user
whether a real ligand, renumbered in its SDF, decodes differently.

This script answers that directly. It runs pose_localmap.py once in the released
configuration and once per permutation seed, maps every permuted prediction back
to the original atom indices, and compares:

  * per-edge bond labels on the original index pairs
  * per-atom formal charge q and hydrogen count h
  * native molecular identity (the built SMILES)
  * constructibility

A molecule whose canonical SMILES is unchanged can still have q or h assigned to
different atoms; that matters downstream, so it is reported separately.

    # released order (baseline)
    python pose_localmap.py --sdf <sdf> --reference_from_sdf --joint_ckpt <ckpt> \
        --outdir <dir> --tag perm0 --limit 100
    # permuted replicates
    python pose_localmap.py --sdf <sdf> --reference_from_sdf --joint_ckpt <ckpt> \
        --outdir <dir> --tag perm1 --limit 100 --permute_seed 1
    ...
    python atom_order_invariance.py --runs <dir>/pose_localmap_perm0.json \
        <dir>/pose_localmap_perm1.json ... --output <json>

Standard library only.
"""
import argparse
import collections
import json
from pathlib import Path


def load(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return {r['name']: r for r in d['records']}


def unpermute(rec):
    """Map a record's atom-indexed fields back to the original SDF numbering.

    permutation[i] is the ORIGINAL index of the atom that featurisation saw at
    position i, so position i maps to original index permutation[i].
    """
    perm = rec.get('permutation')
    pairs = [tuple(sorted(e)) for e in rec['pairs']]
    labels = list(rec['localmap_labels'])
    if perm is None:
        edges = {e: c for e, c in zip(pairs, labels)}
        states = rec.get('localmap_states')
        atoms = {i: s for i, s in enumerate(states)} if states else None
        return edges, atoms
    edges = {tuple(sorted((perm[i], perm[j]))): c for (i, j), c in zip(pairs, labels)}
    states = rec.get('localmap_states')
    atoms = {perm[i]: s for i, s in enumerate(states)} if states else None
    return edges, atoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True,
                    help='first run is the released-order baseline')
    ap.add_argument('--output', default='')
    a = ap.parse_args()

    runs = [load(p) for p in a.runs]
    base = runs[0]
    names = sorted(set(base).intersection(*[set(r) for r in runs[1:]]))
    if not names:
        raise SystemExit('no molecules shared by all runs')
    if any(base[n].get('permutation') is not None for n in names):
        raise SystemExit('the first run must be the unpermuted baseline')

    changed_bonds, changed_states, changed_identity, changed_build = (
        set(), set(), set(), set())
    have_states = False
    for n in names:
        b_edges, b_atoms = unpermute(base[n])
        have_states = have_states or b_atoms is not None
        for r in runs[1:]:
            e, at = unpermute(r[n])
            if e != b_edges:
                changed_bonds.add(n)
            if b_atoms is not None and at is not None and at != b_atoms:
                changed_states.add(n)
            if r[n].get('localmap_smiles') != base[n].get('localmap_smiles'):
                changed_identity.add(n)
            if bool(r[n].get('localmap_valid')) != bool(base[n].get('localmap_valid')):
                changed_build.add(n)

    k, n = len(runs) - 1, len(names)
    print(f'{n} molecules, {k} permuted replicate(s) per molecule\n')
    rows = [('bond labels', changed_bonds),
            ('native identity (SMILES)', changed_identity),
            ('constructibility', changed_build)]
    if have_states:
        rows.insert(1, ('per-atom (q,h) placement', changed_states))
    else:
        print('note: per-atom states are absent from these records, so the '
              '(q,h) placement check was skipped. Re-run the decode with a build '
              'that records localmap_states to enable it.\n')
    for label, s in rows:
        print(f'  {label:28s} changed in {len(s):4d}/{n}  ({100*len(s)/n:5.2f}%)')
    only_states = (changed_states - changed_identity) if have_states else set()
    if only_states:
        print(f'\n  {len(only_states)} molecules keep the same SMILES but place '
              f'q/h on different atoms:')
        print('   ', ', '.join(sorted(only_states)[:25]))
    out = {'runs': list(a.runs), 'n_molecules': n, 'n_permutations': k,
           'changed_bond_labels': sorted(changed_bonds),
           'changed_atom_states': sorted(changed_states),
           'changed_identity': sorted(changed_identity),
           'changed_constructibility': sorted(changed_build),
           'same_identity_different_state_placement': sorted(only_states)}
    if a.output:
        Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
        print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
