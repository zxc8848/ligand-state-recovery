import itertools
import unittest

import numpy as np

from pilot import MASKS, top_by_charge


class ChargeDecoderTests(unittest.TestCase):
    def test_charge_assignments_maximize_the_same_joint_scores(self):
        rng = np.random.default_rng(774)
        for n in [1, 2, 3]:
            logits = rng.normal(size=(3, n, 25)).astype(np.float32)
            proposals = top_by_charge(logits)
            choices = np.array(list(itertools.product(range(25), repeat=n)))
            totals = (choices//5-2).sum(axis=1)
            for row in range(3):
                exhaustive = logits[row, np.arange(n), choices].sum(axis=1)
                for qi, q in enumerate(MASKS):
                    states = proposals[row, qi]
                    self.assertEqual(int((states//5-2).sum()), q)
                    self.assertAlmostEqual(float(logits[row, np.arange(n), states].sum()),
                                           float(exhaustive[totals==q].max()), places=5)


if __name__ == '__main__':
    unittest.main()
