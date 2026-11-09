"""Synthetic chemical, fixed-denominator and statistics regression tests."""
import json
import tempfile
import unittest
from pathlib import Path
from rdkit import Chem, RDLogger
from evaluation_common import aromatic_component_align, score_dump
from paired_verdicts import compare


def graph(mol):
    return {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))):
            4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in mol.GetBonds()}


class EvaluationTests(unittest.TestCase):
    def test_aromatic_forms(self):
        for smiles in ('c1ccccc1', 'c1ccc2ccccc2c1', 'c1cc[nH]c1'):
            mol = Chem.MolFromSmiles(smiles)
            arom = graph(mol)
            k = Chem.Mol(mol); Chem.Kekulize(k, clearAromaticFlags=True)
            kek = graph(k)
            self.assertEqual(aromatic_component_align(kek, arom, mol), arom)
            self.assertEqual(aromatic_component_align(arom, kek, mol), kek)
        mol = Chem.MolFromSmiles('c1ccccc1')
        k = Chem.Mol(mol); Chem.Kekulize(k, clearAromaticFlags=True)
        kek = graph(k)
        self.assertEqual(aromatic_component_align({e: 3-v for e, v in kek.items()}, kek, mol), kek)

    def test_invalid_forms(self):
        mol = Chem.MolFromSmiles('C1=CC=CCC1')
        truth = graph(mol); bad = {e: 4 for e in truth}
        self.assertNotEqual(aromatic_component_align(bad, truth, mol), truth)
        mol = Chem.MolFromSmiles('c1ccccc1'); truth = graph(mol)
        for bad in ({e: 1 for e in truth},
                    {e: (1 if i == 0 else 4) for i, e in enumerate(truth)},
                    dict(list(truth.items())[1:]),
                    {e: (2 if i < 3 else 1) for i, e in enumerate(truth)}):
            self.assertNotEqual(aromatic_component_align(bad, truth, mol), truth)

    def test_fixed_denominator_and_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td); sdf = d/'refs.sdf'
            writer = Chem.SDWriter(str(sdf))
            for name in ('a', 'b'):
                mol = Chem.MolFromSmiles('CC'); mol.SetProp('_Name', name)
                mol.AddConformer(Chem.Conformer(mol.GetNumAtoms())); writer.write(mol)
            mol = Chem.MolFromSmiles('B'); mol.SetProp('_Name', 'excluded_boron')
            mol.AddConformer(Chem.Conformer(mol.GetNumAtoms())); writer.write(mol)
            writer.close()
            edge = d/'edges.jsonl'
            edge.write_text(json.dumps({'name': 'a', 'i': 0, 'j': 1, 'true': 1, 'pred': 1})+'\n')
            old = score_dump(edge, sdf, expected_n=2)
            self.assertEqual(old['n'], 2)
            self.assertEqual(old['raw_n'], 3)
            self.assertEqual(old['excluded_by_element'], ['excluded_boron'])
            with self.assertRaises(ValueError): score_dump(edge, sdf, expected_n=3)
            from population_manifest import population
            from evaluation_common import ALLOWED
            pop = d/'population.json'; pop.write_text(json.dumps(population(sdf, ALLOWED)))
            self.assertEqual(score_dump(edge, sdf, expected_n=2, population_manifest=pop)['n'], 2)
            self.assertEqual(old['missing_predictions'], ['b'])
            self.assertEqual(old['unified_resonance'], .5)
            op = d/'old.json'; op.write_text(json.dumps(old))
            edge.write_text(edge.read_text()+json.dumps({'name': 'b', 'i': 0, 'j': 1, 'true': 1, 'pred': 1})+'\n')
            new = score_dump(edge, sdf, expected_n=2)
            np = d/'new.json'; np.write_text(json.dumps(new))
            result = compare(op, np, sdf, reps=100, expected_n=2)
            self.assertEqual(result['n_molecules'], 2)
            self.assertEqual(result['endpoints']['resonance']['delta'], .5)
            self.assertEqual(result['n_identities'], 1)
            new['per_molecule'].pop(); new['n'] = 1; np.write_text(json.dumps(new))
            with self.assertRaises(ValueError): compare(op, np, sdf, reps=100)


if __name__ == '__main__':
    RDLogger.DisableLog('rdApp.*')
    unittest.main()
