import unittest
from dataclasses import replace
import numpy as np
import spring_window_comparison as s

class TestRatioSelection(unittest.TestCase):
    def h(self, mse, extra): return dict(full_window_mse=mse,extra_count=extra)
    def test_warmup_and_cap(self):
        self.assertEqual(s.ratio_extra_budget(7,13,1,[])[0],7)
        self.assertEqual(s.ratio_extra_budget(7,3,1,[self.h(1,7)])[0],3)
    def test_reference_uses_two_windows(self):
        n,log=s.ratio_extra_budget(7,30,6,[self.h(1,5),self.h(3,5)])
        self.assertEqual(n,15);self.assertEqual(log['reference_mse'],2)
    def test_round_half_up(self):
        self.assertEqual(s.ratio_extra_budget(7,30,1,[self.h(2,5),self.h(2,5)])[0],3)
    def test_zero_cases(self):
        self.assertEqual(s.ratio_extra_budget(7,13,1,[self.h(0,2),self.h(0,2)])[0],13)
        self.assertEqual(s.ratio_extra_budget(7,13,0,[self.h(0,2),self.h(0,2)])[0],2)
        self.assertEqual(s.ratio_extra_budget(7,13,1,[self.h(2,0),self.h(2,0)])[0],1)
    def test_truncated_capacity(self):
        self.assertEqual(s.ratio_extra_budget(4,6,100,[self.h(1,10),self.h(1,10)])[0],6)
    def test_even_baseline_and_logs(self):
        cfg=s.Config(); t,truth,obs,sigma=s.make_data(cfg)
        x,P=s.correct_filter(np.array([1.,0.]),cfg.initial_state_variance*np.eye(2),obs[0],sigma[0]**2)
        counts={k:0 for k in ['simulations','propagated_output_states','span_matrix_products','base_covariance_trials']}
        indices,info=s.select_points(0,20,x,P,[.5,.5],[.5,.5],obs,sigma,cfg,counts)
        self.assertEqual(info['extra_count'],min(info['baseline_count'],20-info['baseline_count']))
        self.assertEqual(indices,sorted(set(indices)));self.assertEqual(indices[-1],20);self.assertNotIn(0,indices)
        self.assertLessEqual(np.ptp(np.diff(info['initial_even_base_points'])),1)
        self.assertEqual(sum(info['extra_allocation']),info['extra_count'])
        self.assertTrue(all(v['max_position_variance']<=cfg.max_position_std**2+1e-12 for v in info['selected_gap_checks']))
        np.testing.assert_allclose(info['full_window_mse'],np.mean(info['diagnostic_point_mse']))
if __name__=='__main__':unittest.main()
