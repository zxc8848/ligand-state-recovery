"""Write Knodle PDBs from only atomic numbers and coordinates, never reference states."""

import argparse
import json
from pathlib import Path

from rdkit import Chem


def geometry_pdb(reference, name):
    if not reference.GetNumConformers() or any(a.GetAtomicNum() < 2 for a in reference.GetAtoms()):
        raise ValueError(f'{name}: expected a heavy-only coordinate inventory')
    atoms = Chem.RWMol()
    source = reference.GetConformer()
    conf = Chem.Conformer(reference.GetNumAtoms())
    conf.Set3D(True)
    for i, a in enumerate(reference.GetAtoms()):
        atoms.AddAtom(Chem.Atom(a.GetAtomicNum()))
        conf.SetAtomPosition(i, source.GetAtomPosition(i))
    atoms.AddConformer(conf, assignId=True)
    atoms.SetProp('_Name', name)
    atoms.UpdatePropertyCache(strict=False)
    block = Chem.MolToPDBBlock(atoms, flavor=10)
    records = [line for line in block.splitlines() if line.startswith(('ATOM  ', 'HETATM'))]
    if len(records) != reference.GetNumAtoms() or any(line[78:80].strip() for line in records):
        raise ValueError(f'{name}: invalid geometry-only export or nonblank atom-charge field')
    if any(line.startswith('CONECT') for line in block.splitlines()):
        raise ValueError(f'{name}: unexpected CONECT in geometry-only PDB')
    return block


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sdf', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--historical-pdb-dir', type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    old_files = {} if not args.historical_pdb_dir else {
        f.stem.split('_', 1)[1]: f for f in args.historical_pdb_dir.glob('*.pdb')}
    stats = dict(old_compared=0, old_with_charge_ids=[], old_charge_atoms=0,
                 old_with_charge_net_neutral_ids=[], changed_pdb_input_ids=[])
    manifest = {'sdf': args.sdf.name, 'records': []}
    names = set()
    for i, mol in enumerate(Chem.SDMolSupplier(str(args.sdf), removeHs=False)):
        if mol is None:
            raise ValueError(f'Unparseable reference {i}')
        name = mol.GetProp('_Name')
        if name in names or len(name) != 4 or not name.isascii() or not name.isalnum():
            raise ValueError(f'Unsafe or duplicate molecule ID: {name!r}')
        names.add(name)
        stem = f'{i:05d}_{name}'
        block = geometry_pdb(mol, name)
        (args.output/f'{stem}.pdb').write_text(block, encoding='ascii')
        manifest['records'].append(dict(stem=stem, mol_index=i, name=name, n_heavy=mol.GetNumAtoms(),
                                        elements=[a.GetAtomicNum() for a in mol.GetAtoms()]))
        if args.historical_pdb_dir:
            old = old_files[name].read_text(encoding='ascii')
            stats['old_compared'] += 1
            charges = sum(bool(line[78:80].strip()) for line in old.splitlines()
                          if line.startswith(('ATOM  ', 'HETATM')))
            if charges:
                stats['old_with_charge_ids'].append(name)
                stats['old_charge_atoms'] += charges
                if Chem.GetFormalCharge(mol) == 0:
                    stats['old_with_charge_net_neutral_ids'].append(name)
            if old != block:
                stats['changed_pdb_input_ids'].append(name)
    stats['n'] = len(names)
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    (args.output/'input_audit.json').write_text(json.dumps(stats, indent=2)+'\n', encoding='utf-8')
    print({k: len(v) if isinstance(v, list) else v for k, v in stats.items()})


if __name__ == '__main__':
    main()
