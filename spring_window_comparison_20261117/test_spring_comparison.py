import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
import spring_window_comparison as s

class TestSharedMechanisms(unittest.TestCase):
    def setUp(self):
        self.cfg=s.Config(duration=.8,window=4,minimum_window=2,maximum_window=6,
                          particles=32,pilot_samples=40,evidence_samples=512,abc_eps_floor=.5)
        self.t,self.truth,self.obs,self.sigma=s.make_data(self.cfg)
        self.counts={k:0 for k in ['simulations','propagated_output_states','span_matrix_products']}
    def test_balanced_tolerance_does_not_exclude_whole_pilot_model(self):
        def d(g,model,*args):
            return np.linspace(1.,2.,len(g)) if model=='Euler' else np.linspace(.01,.02,len(g))
        priors=[s.Prior(self.cfg),s.Prior(self.cfg)]
        with patch.object(s,'distance',d):
            balanced=s.joint_eps(priors,[1,0],[1],np.zeros((1,2)),replace(self.cfg,abc_eps_floor=.001),np.random.default_rng(1),self.counts)
            pooled=s.joint_eps(priors,[1,0],[1],np.zeros((1,2)),replace(self.cfg,abc_eps_floor=.001,abc_tolerance_policy='pooled'),np.random.default_rng(1),self.counts)
        np.testing.assert_allclose(balanced,[1.5,1.3,1.2])
        self.assertLess(pooled[-1],1.)
        self.assertTrue(np.all(np.diff(balanced)<=0))

    def test_normalized_priors(self):
        for p in [s.Prior(self.cfg),s.Prior(self.cfg,[.02,.5,.98],[.2,.5,.3],.03)]:
            area=quad(lambda g:np.exp(p.logpdf(g))[0],.01,.99,epsabs=1e-9)[0]
            self.assertAlmostEqual(area,1.,places=8)
            samples=p.sample(np.random.default_rng(3),10000)
            self.assertTrue(np.all((samples>=.01)&(samples<=.99)))
    def test_direct_merge_same_numerical_models(self):
        for m in s.MODELS:
            full=s.simulate([.2,.6],m,[1.,0.],[1]*8,self.cfg)
            jump=s.simulate([.2,.6],m,[1.,0.],[3,5],self.cfg)
            np.testing.assert_allclose(jump,full[:,[2,7],:],atol=1e-13)
    def test_filter_noise_accumulation(self):
        for m in s.MODELS:
            x=np.array([1.,0.]); P=np.eye(2)*.1
            a,b=s.propagate_filter(x,P,.3,m,7,self.cfg)
            for _ in range(7): x,P=s.propagate_filter(x,P,.3,m,1,self.cfg)
            np.testing.assert_allclose(a,x,atol=1e-13); np.testing.assert_allclose(b,P,atol=1e-13)
    def test_laplace_exact_quadratic_and_prior_curvature(self):
        cfg=self.cfg; prior=s.Prior(cfg,[.4],[1.],.1)
        # Gaussian likelihood with precision 200 and mode .6; posterior precision300.
        def ll(g,*args): return -100*(np.atleast_1d(g)-.6)**2
        with patch.object(s,'gaussian_loglik',ll):
            out=s.laplace_update('Euler',prior,[1,0],[1],np.zeros((1,2)),np.ones(1),cfg,self.counts)
        expected=(200*.6+100*.4)/300
        self.assertAlmostEqual(out['gamma'],expected,places=6)
        self.assertAlmostEqual(out['diagnostics']['curvature'],300.,places=3)
        exact=quad(lambda g:np.exp(ll(g)[0]+prior.logpdf(g)[0]),cfg.gamma_lower,cfg.gamma_upper,epsabs=1e-12)[0]
        self.assertAlmostEqual(np.exp(out['log_score']),exact,places=6)
    def test_boundary_laplace_rejected(self):
        with patch.object(s,'gaussian_loglik',lambda g,*args:-1000*(np.atleast_1d(g)-2)**2):
            with self.assertRaisesRegex(s.InferenceFailure,'boundary'):
                s.laplace_update('Euler',s.Prior(self.cfg),[1,0],[1],np.zeros((1,2)),np.ones(1),self.cfg,self.counts)
    def test_abc_score_vs_exact_tolerance_integral(self):
        cfg=replace(self.cfg,particles=128,evidence_samples=12000)
        prior=s.Prior(cfg,[.4],[1.],.15)
        # Known interval ABC target, preserving real proposal / SMC code.
        def dist(g,*args): return abs(np.atleast_1d(g)-.5)
        with patch.object(s,'distance',dist):
            out=s.abc_update('Euler',prior,[1,0],[1],np.zeros((1,2)),[.3,.2,.1],cfg,np.random.default_rng(4),self.counts)
        exact=quad(lambda g:np.exp(prior.logpdf(g)[0]),.4,.6)[0]
        error=abs(np.exp(out['log_score'])-exact)
        self.assertLess(error,5*out['diagnostics']['evidence_mc_se']+.003)
        self.assertTrue(np.all(np.abs(np.array(out['diagnostics']['particles'])-.5)<=.1))
    def test_full_selection_equivalence_each_backend(self):
        cfg=self.cfg
        original_budget=s.ratio_extra_budget
        self.addCleanup(setattr,s,'ratio_extra_budget',original_budget)
        s.ratio_extra_budget=lambda baseline,capacity,mse,history: (capacity,dict(rule='test_full_selection'))
        for backend in ['laplace','abc']:
            a=s.run_arm(cfg,self.t,self.obs,self.sigma,backend,'fixed_selected')
            b=s.run_arm(cfg,self.t,self.obs,self.sigma,backend,'fixed_all')
            np.testing.assert_allclose(a['model_posterior'],b['model_posterior'],atol=1e-12)
            for w,v in zip(a['windows'],b['windows']):
                np.testing.assert_allclose(w['gamma'],v['gamma'],atol=1e-12)
                np.testing.assert_allclose(w['state_posterior_mean'],v['state_posterior_mean'],atol=1e-12)
    def test_posterior_carry_and_no_duplicate_boundaries(self):
        for backend in ['laplace','abc']:
            a=s.run_arm(self.cfg,self.t,self.obs,self.sigma,backend,'fixed_all')
            self.assertEqual(a['selected_indices'],list(range(len(self.t))))
            for old,new in zip(a['windows'][:-1],a['windows'][1:]):
                self.assertEqual(old['parameter_posterior'],new['parameter_prior'])
                np.testing.assert_allclose(old['model_posterior'],new['model_prior'],atol=1e-13)
                np.testing.assert_array_equal(old['state_posterior_mean'],new['state_prior_mean'])
                np.testing.assert_array_equal(old['state_posterior_covariance'],new['state_prior_covariance'])
                self.assertGreaterEqual(np.linalg.eigvalsh(new['state_posterior_covariance']).min(),-1e-12)
    def test_evidence_update_identity(self):
        a=s.run_arm(self.cfg,self.t,self.obs,self.sigma,'laplace','fixed_all')
        for w in a['windows']:
            lp=np.log(w['model_prior'])+w['log_score']; lp-=logsumexp(lp)
            np.testing.assert_allclose(np.exp(lp),w['model_posterior'],atol=1e-13)
    def test_random_stream_reproducibility(self):
        a=s.run_arm(self.cfg,self.t,self.obs,self.sigma,'abc','fixed_all')
        b=s.run_arm(self.cfg,self.t,self.obs,self.sigma,'abc','fixed_all')
        self.assertEqual(a['windows'],b['windows'])
    def test_zero_acceptance_raises(self):
        with patch.object(s,'distance',lambda g,*args:np.full(len(g),100.)):
            with self.assertRaises(s.InferenceFailure):
                s.abc_update('Euler',s.Prior(self.cfg),[1,0],[1],np.zeros((1,2)),[.1],replace(self.cfg,max_attempts=20),np.random.default_rng(1),self.counts)
    def test_adaptive_rule_direction(self):
        cfg=self.cfg; y=np.array([0.,1.])
        w,_=s.next_window(y,y+.5,4,.1,cfg); self.assertEqual(w,2)
        w,_=s.next_window(y,y+.01,4,.1,cfg); self.assertEqual(w,6)
    def test_locked_adaptive_matches_fixed_all(self):
        cfg=replace(self.cfg,minimum_window=4,maximum_window=4)
        for backend in ['laplace','abc']:
            a=s.run_arm(cfg,self.t,self.obs,self.sigma,backend,'adaptive_all')
            b=s.run_arm(cfg,self.t,self.obs,self.sigma,backend,'fixed_all')
            np.testing.assert_array_equal(a['model_posterior'],b['model_posterior'])
            for w,v in zip(a['windows'],b['windows']):
                self.assertEqual(w['parameter_posterior'],v['parameter_posterior'])
                self.assertEqual(w['state_posterior_mean'],v['state_posterior_mean'])
    def test_frozen_selection_does_not_mutate(self):
        x=np.array([1.,0.]); P=np.eye(2)*.0001
        estimates=np.array([.4,.3]); weights=np.array([.4,.6])
        snapshots=[a.copy() for a in [x,P,estimates,weights]]
        counts=dict(simulations=0,propagated_output_states=0,span_matrix_products=0,base_covariance_trials=0)
        s.select_points(0,4,x,P,estimates,weights,self.obs,self.sigma,self.cfg,counts)
        for a,b in zip([x,P,estimates,weights],snapshots): np.testing.assert_array_equal(a,b)
    def test_infeasible_bound_is_explicit(self):
        cfg=replace(self.cfg,max_position_std=.0001)
        with self.assertRaises(s.InferenceFailure):
            s.run_arm(cfg,self.t,self.obs,self.sigma,'laplace','fixed_selected')

if __name__=='__main__': unittest.main()
