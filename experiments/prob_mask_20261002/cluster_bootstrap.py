"""Identity and molecule bootstrap helpers; CLI consumes versioned unified flags.

This clusters molecular identity, not Bemis-Murcko scaffold. Both percentile
intervals are retained; exact McNemar is a molecule-level descriptive test.
"""
import argparse
import json
import math
import random
from pathlib import Path


def identity_key(mol):
    from rdkit import Chem
    return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(mol)), isomericSmiles=False)


def mcnemar(pairs):
    b = sum(1 for x, y in pairs if y and not x)
    c = sum(1 for x, y in pairs if x and not y)
    n = b+c
    if n == 0: return 1.
    return min(1., sum(math.comb(n, i) for i in range(min(b, c)+1))/2**n*2)


def boot_molecules(pairs, reps, seed):
    if not pairs or reps < 40: raise ValueError('Need nonempty pairs and >=40 replicates')
    rng = random.Random(seed); n = len(pairs); ds = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        ds.append(sum(y-x for x, y in s)/n)
    ds.sort()
    return ds[int(.025*reps)], ds[min(reps-1, int(.975*reps))]


def boot_clusters(clusters, reps, seed):
    if not clusters or reps < 40: raise ValueError('Need nonempty clusters and >=40 replicates')
    rng = random.Random(seed); k = len(clusters); ds = []
    for _ in range(reps):
        drawn = [clusters[rng.randrange(k)] for _ in range(k)]
        n = sum(len(c) for c in drawn)
        ds.append(sum(y-x for c in drawn for x, y in c)/n)
    ds.sort()
    return ds[int(.025*reps)], ds[min(reps-1, int(.975*reps))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_sdf', required=True)
    ap.add_argument('--old_species', '--old_verdicts', dest='old_species', required=True)
    ap.add_argument('--new_species', '--new_verdicts', dest='new_species', required=True)
    ap.add_argument('--old_per_edge', help='deprecated; not used to derive flags')
    ap.add_argument('--new_per_edge', help='deprecated; not used to derive flags')
    ap.add_argument('--reps', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--label', default='')
    ap.add_argument('--output', default='')
    a = ap.parse_args()
    from paired_verdicts import compare
    out = compare(a.old_species, a.new_species, a.reference_sdf, a.reps, a.seed)
    out['label'] = a.label
    if a.output: Path(a.output).write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps(out, indent=1))


if __name__ == '__main__': main()
