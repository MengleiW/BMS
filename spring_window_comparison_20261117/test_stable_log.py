import unittest
from decimal import Decimal
import numpy as np
from spring_window_comparison import stable_binary_log_posterior, decimal_probability

class StableLogTests(unittest.TestCase):
    def test_equal_scores(self):
        np.testing.assert_allclose(stable_binary_log_posterior([0., 0.]), [-np.log(2)]*2)

    def test_near_one_is_preserved(self):
        lp=stable_binary_log_posterior([0.,158.])
        self.assertLess(lp[1],0.)
        self.assertLess(abs(np.logaddexp(*lp)),1e-14)
        p=Decimal(decimal_probability(lp[1]))
        self.assertTrue(Decimal(0)<p<Decimal(1))
        np.testing.assert_allclose(stable_binary_log_posterior([158.,0.]),lp[::-1])

    def test_underflow_is_disclosed(self):
        lp=stable_binary_log_posterior([0.,800.])
        self.assertEqual(lp[1],0.)
        self.assertIn('underflow',decimal_probability(lp[1]))

if __name__=='__main__': unittest.main()
