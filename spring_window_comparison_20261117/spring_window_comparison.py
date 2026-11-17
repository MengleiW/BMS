"""Spring-mass window comparison with proportional extra-observation selection.

Shared Laplace / ABC-SMC / EKF mechanisms are preserved across window policies.
See README.md for the attachment rule, implementation conventions and limitations.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse, json, time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
from scipy.special import ndtr, ndtri, logsumexp

MODELS = ('Euler', 'Heun')

@dataclass
class Config:
    seed: int = 42
    dt: float = .1
    duration: float = 15.
    gamma_true: float = .3
    stiffness: float = 1.
    noise_low: float = .01
    # Noise standard deviation at t >= 12 s; change back to 0.1 to restore the noise jump.
    noise_high: float = .01
    noise_switch_time: float = 12.
    # Historical experiment knowingly assumes constant low noise despite jump.
    # Both arms use the identical setting. Enable known noise for a control.
    known_noise_schedule: bool = False
    window: int = 20
    minimum_window: int = 4
    maximum_window: int = 40
    window_step: int = 2
    relative_error_tolerance: float = .05
    gamma_lower: float = .01
    gamma_upper: float = .99
    initial_state_variance: float = .1
    process_variance: float = .01  # per original dt, not per selected gap
    max_position_std: float = .2  # conditional on incoming parameter estimates
    segments: int = 5
    # Legacy CLI/config field retained for compatibility; ratio rule determines extras.
    extra_points: int = 4
    particles: int = 64
    pilot_samples: int = 100
    evidence_samples: int = 1024
    max_attempts: int = 30000
    abc_eps_floor: float = .1
    abc_tolerance_policy: str = "balanced"  # pooled reproduces historical failure
    prior_odds_cap: float | None = None

class InferenceFailure(RuntimeError):
    pass


def transition(gamma, model, cfg):
    """Original base-grid Euler or explicit trapezoidal (Heun) matrix."""
    g = np.atleast_1d(gamma)
    A = np.zeros((len(g),2,2)); A[:,0,1]=1
    A[:,1,0]=-cfg.stiffness; A[:,1,1]=-g
    B = np.eye(2)+cfg.dt*A
    if model == 'Heun': B += .5*cfg.dt**2*(A@A)
    elif model != 'Euler': raise ValueError(model)
    return B


def simulate(gamma, model, x0, gaps, cfg, counts=None):
    """No interior state outputs. Compose base-grid matrices, NOT a big Euler step.

    This preserves the candidate numerical model when observations are omitted.
    Construct needed span matrices once per parameter batch; reuse within it.
    """
    B = transition(gamma,model,cfg)
    x = np.broadcast_to(x0,(len(B),2)).copy()
    powers={0:np.broadcast_to(np.eye(2),B.shape).copy(),1:B}
    for d in range(2,max(gaps,default=1)+1): powers[d]=B@powers[d-1]
    outputs=[]
    for d in gaps:
        x=np.einsum('nij,nj->ni',powers[int(d)],x); outputs.append(x.copy())
    if counts is not None:
        counts['simulations'] += len(B)
        counts['propagated_output_states'] += len(B)*len(gaps)
        counts['span_matrix_products'] += len(B)*max(0,max(gaps,default=1)-1)
    return np.stack(outputs,axis=1) if outputs else np.empty((len(B),0,2))


class Prior:
    """Normalized bounded uniform or truncated-normal mixture parameter prior.

    A Laplace posterior is a one-component mixture; ABC carries all weighted
    particles into a smoothed mixture. Smoothing is explicit, not exact Bayes.
    """
    def __init__(self,cfg,centers=None,weights=None,sd=None):
        self.lo,self.hi=cfg.gamma_lower,cfg.gamma_upper
        self.centers=None if centers is None else np.atleast_1d(centers).astype(float)
        if self.centers is not None:
            self.weights=np.ones(len(self.centers))/len(self.centers) if weights is None else np.asarray(weights,float).copy()
            self.weights/=self.weights.sum()
            self.sd=float(sd)
            if not self.sd>0: raise ValueError('Prior SD must be positive')
            self.a=ndtr((self.lo-self.centers)/self.sd)
            self.z=ndtr((self.hi-self.centers)/self.sd)-self.a
            if np.any(self.z<=0): raise InferenceFailure('Truncated prior has no numerical mass')
    def logpdf(self,g):
        g=np.atleast_1d(g).astype(float)
        if self.centers is None: value=np.full(len(g),-np.log(self.hi-self.lo))
        else:
            v=-.5*((g[:,None]-self.centers)/self.sd)**2-np.log(self.sd*np.sqrt(2*np.pi))-np.log(self.z)+np.log(self.weights)
            value=logsumexp(v,axis=1)
        return np.where((g>=self.lo)&(g<=self.hi),value,-np.inf)
    def sample(self,rng,n):
        if self.centers is None: return rng.uniform(self.lo,self.hi,n)
        i=rng.choice(len(self.centers),size=n,p=self.weights)
        u=self.a[i]+rng.random(n)*self.z[i]
        return self.centers[i]+self.sd*ndtri(np.clip(u,1e-15,1-1e-15))
    def summary(self):
        return dict(kind='uniform' if self.centers is None else 'truncated_normal_mixture',
                    centers=None if self.centers is None else self.centers.tolist(),
                    weights=None if self.centers is None else self.weights.tolist(),
                    sd=None if self.centers is None else self.sd)


def gaussian_loglik(gamma,model,x0,gaps,obs,sigma,cfg,counts=None):
    pred=simulate(gamma,model,x0,gaps,cfg,counts)
    res=(obs[None,:,:]-pred)/sigma[None,:,None]
    return -.5*np.sum(res**2+np.log(2*np.pi*sigma[None,:,None]**2),axis=(1,2))


def laplace_update(model,prior,x0,gaps,obs,sigma,cfg,counts):
    def logpost(g): return gaussian_loglik(g,model,x0,gaps,obs,sigma,cfg,counts)+prior.logpdf(g)
    grid=np.linspace(cfg.gamma_lower,cfg.gamma_upper,81)
    values=logpost(grid); best=int(np.argmax(values))
    left=grid[max(0,best-1)]; right=grid[min(len(grid)-1,best+1)]
    fit=minimize_scalar(lambda g:-float(logpost(g)[0]),bounds=(left,right),method='bounded',options={'xatol':1e-9})
    mode=float(fit.x) if fit.success and -fit.fun>values[best] else float(grid[best])
    # This implementation requires an interior stationary mode. A constrained
    # boundary mode needs a different asymptotic expansion; fail explicitly.
    h=1e-4
    if min(mode-cfg.gamma_lower,cfg.gamma_upper-mode) <= 2*h:
        raise InferenceFailure('Laplace mode at parameter boundary; interior approximation invalid')
    v=logpost([mode-h,mode,mode+h]); precision=-(v[2]-2*v[1]+v[0])/h**2
    if not np.isfinite(precision) or precision<=0:
        raise InferenceFailure('Nonpositive Laplace curvature; no fabricated evidence fallback')
    sd=float(1/np.sqrt(precision)); posterior=Prior(cfg,[mode],[1.],sd)
    # Finite-support Laplace mass correction; boundary/strongly non-Gaussian
    # posterior still needs independent checks, not an accuracy guarantee.
    mass=ndtr((cfg.gamma_upper-mode)/sd)-ndtr((cfg.gamma_lower-mode)/sd)
    logZ=float(logpost(mode)[0]+.5*np.log(2*np.pi/precision)+np.log(mass))
    return dict(gamma=mode,log_score=logZ,prior=posterior,
                diagnostics=dict(curvature=float(precision),near_boundary=bool(min(mode-cfg.gamma_lower,cfg.gamma_upper-mode)<3*sd)))


def distance(gamma,model,x0,gaps,obs,cfg,counts):
    return np.linalg.norm(simulate(gamma,model,x0,gaps,cfg,counts)-obs[None,:,:],axis=(1,2))


def posterior_kernel(cfg,particles,weights):
    mean=np.sum(particles*weights)
    sd=max(.01,.10*np.sqrt(np.sum(weights*(particles-mean)**2)))
    return Prior(cfg,particles,weights,sd)


def joint_eps(priors,x0,gaps,obs,cfg,rng,counts):
    # Same thresholds for Euler/Heun in this window, using their proper priors.
    # These data-adapted tolerances are not model-independent evidence calibration.
    values=[]
    for m,prior in zip(MODELS,priors):
        g=prior.sample(rng,cfg.pilot_samples//2)
        values.append(distance(g,m,x0,gaps,obs,cfg,counts))
    if cfg.abc_tolerance_policy == 'pooled':
        thresholds=np.percentile(np.concatenate(values),[50,30,20])
    elif cfg.abc_tolerance_policy == 'balanced':
        # A pooled low quantile can be BELOW an entire model's minimum distance.
        # Use a common threshold at which every pilot has the requested fraction.
        # This deliberately changes the ABC target, not just its proposal.
        thresholds=np.max([np.percentile(v,[50,30,20]) for v in values],axis=0)
    else: raise ValueError('Unknown ABC tolerance policy')
    return np.maximum(thresholds,cfg.abc_eps_floor)


def abc_update(model,prior,x0,gaps,obs,eps,cfg,rng,counts):
    """ABC-SMC for deterministic distance <= epsilon pseudo-likelihood.

    Target each population: incoming_prior(gamma) * indicator(distance<=eps).
    Importance weight uses normalized incoming_prior / actual proposal density.
    Not the erroneous historical product of mean accepted weights.
    """
    proposal=prior; generations=[]
    def draw(n):
        g=prior.sample(rng,n); use_kernel=rng.random(n)>=.5
        g[use_kernel]=proposal.sample(rng,int(use_kernel.sum()))
        return g
    def logproposal(g):
        return np.logaddexp(prior.logpdf(g),proposal.logpdf(g))-np.log(2.)
    for tolerance in eps:
        accepted=[]; attempts=0
        while sum(len(a) for a in accepted)<cfg.particles and attempts<cfg.max_attempts:
            n=min(128,cfg.max_attempts-attempts)
            g=draw(n); d=distance(g,model,x0,gaps,obs,cfg,counts)
            accepted.append(g[d<=tolerance]); attempts+=n
        joined=np.concatenate(accepted) if accepted else np.array([])
        if len(joined)<cfg.particles:
            raise InferenceFailure(f'ABC {model}: only {len(joined)} accepted in {attempts} proposals, eps={tolerance:g}')
        particles=joined[:cfg.particles]
        logweights=prior.logpdf(particles)-logproposal(particles)
        weights=np.exp(logweights-logsumexp(logweights))
        proposal=posterior_kernel(cfg,particles,weights)
        generations.append(dict(epsilon=float(tolerance),proposals=attempts,ess=float(1/np.sum(weights**2))))
    # Independent, fixed-count importance sampling from a defensive mixture.
    # Unlike averages over accepted proposals, this includes rejected draws.
    n=cfg.evidence_samples; choose=rng.random(n)<.5
    g=prior.sample(rng,n); g[~choose]=proposal.sample(rng,int((~choose).sum()))
    logp=prior.logpdf(g)
    logq=np.logaddexp(logp,proposal.logpdf(g))-np.log(2.)
    terms=np.exp(logp-logq)*(distance(g,model,x0,gaps,obs,cfg,counts)<=eps[-1])
    score=float(terms.mean()); se=float(terms.std(ddof=1)/np.sqrt(n))
    if not score>0: raise InferenceFailure('ABC evidence estimate zero; increase evidence_samples or tolerance')
    return dict(gamma=float(np.sum(weights*particles)),log_score=float(np.log(score)),prior=proposal,
                diagnostics=dict(generations=generations,evidence_score=score,evidence_mc_se=se,
                    particles=particles.tolist(),weights=weights.tolist()))


def propagate_filter(x,P,gamma,model,gap,cfg,counts=None):
    B=transition([gamma],model,cfg)[0]; F=np.eye(2); Q=np.zeros((2,2))
    for _ in range(gap): F=B@F; Q=B@Q@B.T+cfg.process_variance*np.eye(2)
    if counts is not None: counts['filter_span_matrix_products']+=3*gap
    return F@x,F@P@F.T+Q


def correct_filter(x,P,value,variance):
    R=variance*np.eye(2); K=np.linalg.solve((P+R).T,P.T).T
    residual=value-x; J=np.eye(2)-K
    return x+K@residual,J@P@J.T+K@R@K.T


def state_update(x,P,estimates,model_weights,indices,start,obs,sigma,cfg,counts):
    # Separate filters, never augment the state with gamma. Both methods have
    # their own state track; Laplace and ABC evidence are never mixed.
    means=[]; covs=[]
    for model,gamma in zip(MODELS,estimates):
        mean=x.copy(); cov=P.copy(); prev=start
        for idx in indices:
            mean,cov=propagate_filter(mean,cov,gamma,model,idx-prev,cfg,counts)
            mean,cov=correct_filter(mean,cov,obs[idx],sigma[idx]**2)
            counts['filter_corrections']+=1; prev=idx
        means.append(mean); covs.append(cov)
    mean=np.sum(np.asarray(means)*model_weights[:,None],axis=0)
    cov=sum(w*(p+np.outer(m-mean,m-mean)) for w,m,p in zip(model_weights,means,covs))
    return mean,cov


def ratio_extra_budget(baseline_count, capacity, current_mse, history):
    """Attachment ratio rule, with explicit finite-grid implementation conventions."""
    if len(history)<2:
        return min(baseline_count,capacity), dict(rule='warmup_twice_baseline',reference_mse=None,
            residual_ratio=None,previous_extra_count=None,raw_extra_count=float(baseline_count))
    reference=float(np.mean([h['full_window_mse'] for h in history[-2:]]))
    previous=int(history[-1]['extra_count'])
    if reference==0:
        ratio=None if current_mse>0 else 1.0
        raw=float(capacity) if current_mse>0 else float(previous)
        rule='zero_reference_fill_capacity' if current_mse>0 else 'both_zero_keep_previous'
    else:
        ratio=float(current_mse/reference)
        raw=float(previous)*ratio
        rule='previous_extra_times_mse_ratio'
    if previous==0 and current_mse>0 and capacity>0:
        raw=max(1.0,raw); rule+=';restart_at_least_one'
    # Round half up (not bankers rounding), then cap. Clamp before converting huge values.
    bounded=min(float(capacity),max(0.0,raw))
    budget=min(capacity,int(np.floor(bounded+.5)))
    return budget,dict(rule=rule,reference_mse=reference,residual_ratio=ratio if ratio is None or np.isfinite(ratio) else None,
        previous_extra_count=previous,raw_extra_count=raw if np.isfinite(raw) else None)


def check_base_gaps(base,x,P,estimates,sigma,cfg,counts):
    """Planning covariance only: never assimilate actual observations or change x/P."""
    covs=[P.copy(),P.copy()]; records=[]
    for left,right in zip(base[:-1],base[1:]):
        for span in range(1,right-left+1):
            trial=[propagate_filter(x,p,g,m,span,cfg)[1]
                for p,g,m in zip(covs,estimates,MODELS)]
            counts['base_covariance_trials']+=2
            var=max(float(p[0,0]) for p in trial)
            if var>cfg.max_position_std**2+1e-12:
                return False,records,dict(left=int(left),right=int(right),first_unsafe_span=span)
        records.append(dict(start=int(left),end=int(right),gap_steps=int(right-left),
            gap_seconds=float((right-left)*cfg.dt),max_position_variance=var))
        covs=[correct_filter(x,p,x,sigma[right]**2)[1] for p in trial]
    return True,records,None


def select_points(start,end,x,P,estimates,weights,obs,sigma,cfg,counts,history=None):
    history=[] if history is None else history
    candidates=np.arange(start+1,end+1); gaps=np.ones(len(candidates),dtype=int)
    pred=sum(w*simulate([g],m,x,gaps,cfg,counts)[0] for w,g,m in zip(weights,estimates,MODELS))
    # Preserve existing unnormalized equal-weight position/velocity MSE convention.
    score_per_point=np.mean((obs[candidates]-pred)**2,axis=1)
    full_mse=float(np.mean(score_per_point))
    # First estimate a feasible baseline count using the existing uncertainty rule.
    greedy=[start]; covs=[P.copy(),P.copy()]
    while greedy[-1]<end:
        safe=0
        for span in range(1,end-greedy[-1]+1):
            trial=[propagate_filter(x,p,g,m,span,cfg)[1] for p,g,m in zip(covs,estimates,MODELS)]
            counts['base_covariance_trials']+=2
            if max(p[0,0] for p in trial)>cfg.max_position_std**2+1e-12: break
            safe=span
        if safe==0: raise InferenceFailure('State-uncertainty bound infeasible even at one base-grid interval')
        point=greedy[-1]+safe
        covs=[correct_filter(x,propagate_filter(x,p,g,m,safe,cfg)[1],x,sigma[point]**2)[1]
              for p,g,m in zip(covs,estimates,MODELS)]
        greedy.append(point)
    # Spread the same count evenly over the actual uniform observation grid, retaining end.
    base=sorted(set(np.floor(np.linspace(start,end,len(greedy))+.5).astype(int).tolist()))
    initial_even=base.copy(); repairs=[]
    while True:
        valid,gap_records,failure=check_base_gaps(base,x,P,estimates,sigma,cfg,counts)
        if valid: break
        left,right=failure['left'],failure['right']
        if right-left<=1:
            raise InferenceFailure('Available observation grid cannot satisfy uncertainty bound')
        point=left+max(1,failure['first_unsafe_span']-1)
        if point>=right: raise InferenceFailure('Unable to repair baseline gap')
        base.append(point);base.sort();repairs.append(point)
    baseline_count=len(base)-1 # Shared left boundary is initialization, not a new observation.
    capacity_total=(end-start)-baseline_count
    extra_budget,budget_info=ratio_extra_budget(baseline_count,capacity_total,full_mse,history)
    bins=np.array_split(candidates,min(cfg.segments,len(candidates)))
    scores=np.array([np.mean(score_per_point[b-start-1]) for b in bins])
    available=[set(map(int,b))-set(base) for b in bins]
    capacity=np.array([len(a) for a in available]); allocation=np.zeros(len(bins),int)
    left=extra_budget
    while left:
        active=allocation<capacity; w=np.where(active,scores,0.)
        if w.sum()==0: w=active.astype(float)
        q=left*w/w.sum(); remaining=capacity-allocation
        saturated=active&(q>=remaining)
        if np.any(saturated):
            allocation[saturated]+=remaining[saturated]; left-=int(remaining[saturated].sum()); continue
        floor=np.floor(q).astype(int); allocation+=floor; left-=int(floor.sum())
        for j in np.argsort(-(q-floor),kind='stable'):
            if left and allocation[j]<capacity[j]: allocation[j]+=1; left-=1
    selected=set(base)
    for j,n in enumerate(allocation):
        for _ in range(n):
            # Fill largest uncovered gaps; deterministic earliest-point tie break.
            point=max(sorted(available[j]),key=lambda p:min(abs(p-s) for s in selected))
            selected.add(point); available[j].remove(point)
    selected=sorted(selected-{start})
    # Verify final actual gaps too (including across segment boundaries).
    valid,final_gaps,failure=check_base_gaps([start]+selected,x,P,estimates,sigma,cfg,counts)
    if not valid: raise InferenceFailure('Final selected gaps violate uncertainty bound: '+str(failure))
    extra_indices=sorted(set(selected)-set(base))
    info=dict(base_points=base,baseline_count=baseline_count,initial_even_base_points=initial_even,
        baseline_gap_repairs=repairs,baseline_gap_checks=gap_records,selected_gap_checks=final_gaps,
        full_window_mse=full_mse,diagnostic_component_mse=np.mean((obs[candidates]-pred)**2,axis=0).tolist(),
        diagnostic_predictions=pred.tolist(),diagnostic_point_mse=score_per_point.tolist(),
        extra_count=len(extra_indices),extra_indices=extra_indices,selected_count=len(selected),
        available_count=end-start,remaining_capacity=capacity_total,segment_scores=scores.tolist(),
        extra_allocation=allocation.tolist(),**budget_info)
    return selected,info



def next_window(observed,predicted,width,previous,cfg):
    rmse=float(np.sqrt(np.mean((observed-predicted)**2)))
    span=float(np.ptp(observed)); error=rmse/max(span,cfg.noise_low,1e-12)
    new=width
    if previous is not None:
        if error>(1+cfg.relative_error_tolerance)*previous: new=max(cfg.minimum_window,width-cfg.window_step)
        elif error<(1-cfg.relative_error_tolerance)*previous: new=min(cfg.maximum_window,width+cfg.window_step)
    return new,error


def make_data(cfg):
    t=np.arange(int(round(cfg.duration/cfg.dt))+1)*cfg.dt
    truth=solve_ivp(lambda t,y:[y[1],-cfg.gamma_true*y[1]-cfg.stiffness*y[0]],
                    [0,t[-1]], [1.,0.],t_eval=t,rtol=1e-10,atol=1e-12).y.T
    actual_sigma=np.where(t<cfg.noise_switch_time,cfg.noise_low,cfg.noise_high)
    rng=np.random.default_rng(cfg.seed)
    obs=truth+np.column_stack([rng.normal(0,actual_sigma),rng.normal(0,actual_sigma)])
    assumed=actual_sigma.copy() if cfg.known_noise_schedule else np.full(len(t),cfg.noise_low)
    return t,truth,obs,assumed


def stable_binary_log_posterior(raw):
    """Diagnostic representation from raw log scores, not rounded probabilities."""
    d=float(raw[1]-raw[0])
    if not np.isfinite(d): raise InferenceFailure('Nonfinite model log odds')
    small=np.log1p(np.exp(-abs(d)))
    return np.array([-max(d,0.)-small, -max(-d,0.)-small])


def decimal_probability(logp):
    """Decimal display of the finite stored log value; not extra inference precision."""
    from decimal import Decimal, localcontext
    if logp==0: return '1 (log posterior rounded to zero; possible underflow)'
    digits=max(50, int(max(0.,-np.log10(abs(logp))))+25)
    with localcontext() as ctx:
        ctx.prec=digits
        return str(Decimal(str(float(logp))).exp())


def run_arm(cfg,t,obs,sigma,backend,policy):
    if backend not in ('laplace','abc') or policy not in ('adaptive_all','fixed_selected','fixed_all'):
        raise ValueError('Invalid backend/policy')
    priors=[Prior(cfg),Prior(cfg)]; logweights=np.log([.5,.5]); estimates=[.5,.5]
    x,P=correct_filter(np.array([1.,0.]),cfg.initial_state_variance*np.eye(2),obs[0],sigma[0]**2)
    width=cfg.window; previous=None; start=0; windows=[]; selected_all=[0]
    counts={k:0 for k in ['simulations','propagated_output_states','span_matrix_products',
                        'filter_span_matrix_products','filter_corrections','base_covariance_trials']}
    began=time.perf_counter()
    while start<len(t)-1:
        end=min(start+width,len(t)-1); snapshot=[p.summary() for p in priors]
        prior_weights=logweights.copy()
        if cfg.prior_odds_cap is not None:
            bound=np.log(cfg.prior_odds_cap); odds=np.clip(logweights[1]-logweights[0],-bound,bound)
            prior_weights=np.array([0.,odds]); prior_weights-=logsumexp(prior_weights)
        if policy=='fixed_selected':
            indices,selection=select_points(start,end,x,P,estimates,np.exp(prior_weights),obs,sigma,cfg,counts,history=[w['selection'] for w in windows])
        else: indices=list(range(start+1,end+1)); selection={}
        gaps=np.diff([start]+indices); values=obs[indices]; sigmas=sigma[indices]
        # Matching windows receive matching RNG streams; differing schedules
        # cannot consume identical proposals, which is an unavoidable limitation.
        try:
            eps=None
            if backend=='abc': eps=joint_eps(priors,x,gaps,values,cfg,np.random.default_rng(np.random.SeedSequence([cfg.seed,start,0])),counts)
            updates=[]
            for m,model in enumerate(MODELS):
                if backend=='laplace': update=laplace_update(model,priors[m],x,gaps,values,sigmas,cfg,counts)
                else: update=abc_update(model,priors[m],x,gaps,values,eps,cfg,np.random.default_rng(np.random.SeedSequence([cfg.seed,start,m+1])),counts)
                updates.append(update)
        except InferenceFailure as exc:
            return dict(status='failed',reason=str(exc),backend=backend,policy=policy,
                windows=windows,selected_indices=selected_all,counts=counts,
                failed_window=dict(start=int(start),end=int(end),selected=indices),
                seconds=time.perf_counter()-began)
        logscore=np.array([u['log_score'] for u in updates]); logweights=prior_weights+logscore
        stable_logp=stable_binary_log_posterior(logweights)
        logweights-=logsumexp(logweights)
        estimates=[u['gamma'] for u in updates]; priors=[u['prior'] for u in updates]
        # Historical adaptation metric averages absolute per-model position
        # residuals; preserve that definition rather than inventing new BMA metric.
        all_indices=np.arange(start+1,end+1)
        fitted=[simulate([g],m,x,np.ones(len(all_indices),int),cfg,counts)[0] for g,m in zip(estimates,MODELS)] if policy=='adaptive_all' else []
        nextwidth=width; error=None
        if policy=='adaptive_all':
            avg_abs=np.mean([np.abs(obs[all_indices,0]-p[:,0]) for p in fitted],axis=0)
            nextwidth,error=next_window(obs[all_indices,0],obs[all_indices,0]-avg_abs,width,previous,cfg)
        incoming_x=x.copy(); incoming_P=P.copy()
        x,P=state_update(x,P,estimates,np.exp(logweights),indices,start,obs,sigma,cfg,counts)
        windows.append(dict(start=int(start),end=int(end),selected=indices,selection=selection,
            parameter_prior=snapshot,parameter_posterior=[p.summary() for p in priors],
            model_prior=np.exp(prior_weights).tolist(),log_score=logscore.tolist(),model_posterior=np.exp(logweights).tolist(),
            model_log_posterior=stable_logp.tolist(),
            model_log_odds_trapezoidal_over_euler=float(logweights[1]-logweights[0]),
            model_posterior_decimal=[decimal_probability(v) for v in stable_logp],
            gamma=estimates,state_prior_mean=incoming_x.tolist(),state_prior_covariance=incoming_P.tolist(),
            state_posterior_mean=x.tolist(),state_posterior_covariance=P.tolist(),
            adaptation_error=error,epsilon=None if eps is None else eps.tolist(),
            diagnostics=[u['diagnostics'] for u in updates]))
        selected_all.extend(indices); start=end; width=nextwidth; previous=error
    return dict(status='completed',backend=backend,policy=policy,windows=windows,selected_indices=selected_all,
                counts=counts,seconds=time.perf_counter()-began,model_posterior=np.exp(logweights).tolist())


def compare(cfg):
    if not (cfg.dt>0 and cfg.duration>=cfg.dt and cfg.noise_low>0 and cfg.noise_high>0):
        raise ValueError('Invalid physical time or noise')
    if not (1<=cfg.minimum_window<=cfg.window<=cfg.maximum_window and cfg.window_step>=1):
        raise ValueError('Invalid window bounds')
    if cfg.particles<2 or cfg.evidence_samples<2 or cfg.pilot_samples<20 or cfg.max_attempts<cfg.particles:
        raise ValueError('Invalid Monte Carlo budgets')
    if cfg.segments<1 or cfg.extra_points<0 or cfg.max_position_std<=0 or cfg.abc_eps_floor<=0:
        raise ValueError('Invalid point-selection or tolerance settings')
    if cfg.initial_state_variance<0 or cfg.process_variance<0 or cfg.gamma_lower>=cfg.gamma_upper:
        raise ValueError('Invalid prior or covariance')
    if cfg.prior_odds_cap is not None and (not np.isfinite(cfg.prior_odds_cap) or cfg.prior_odds_cap<1):
        raise ValueError('prior_odds_cap must be finite and >=1')
    t,truth,obs,sigma=make_data(cfg); arms={}
    for backend in ['laplace','abc']:
        for policy in ['adaptive_all','fixed_selected','fixed_all']:
            name=f'{backend}_{policy}'
            print('Running',name,flush=True)
            try: arms[name]=run_arm(cfg,t,obs,sigma,backend,policy)
            except InferenceFailure as exc: arms[name]=dict(status='failed',reason=str(exc))
    return dict(config=asdict(cfg),reference='windowedmethod_testing.py; user has not confirmed exact historical version',
        model_names=list(MODELS),times=t.tolist(),truth=truth.tolist(),observations=obs.tolist(),
        assumed_sigma=sigma.tolist(),arms=arms)


def save(result,output):
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist()),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Display only: inference, stored probabilities and random draws are unchanged.
    fig,axs=plt.subplots(2,4,figsize=(26,11),constrained_layout=True)
    policies=['adaptive_all','fixed_selected','fixed_all']
    colors=['tab:blue','tab:orange','tab:green']
    for row,backend in enumerate(['laplace','abc']):
        underflow=[]
        trapezoidal_underflow=[]
        for policy,color,marker in zip(policies,colors,['o','^','s']):
            arm=result['arms'][f'{backend}_{policy}']
            if arm['status']!='completed': continue
            windows=arm['windows']
            ends=[result['times'][w['end']] for w in windows]
            times=[result['times'][0]]+ends
            # Read BOTH probabilities directly; never compute Euler as 1 - Trapezoidal.
            probs=np.array([[0.5,0.5]]+[w['model_posterior'] for w in windows])
            for model,label,style in [(0,'Euler','--'),(1,'Trapezoidal','-')]:
                axs[row,0].plot(times,probs[:,model],marker=marker,markersize=7,markerfacecolor='none',linestyle=style,
                    color=color,label=f'{policy}: {label}')
            euler=probs[:,0]
            positive=euler>0
            axs[row,1].plot(np.array(times)[positive],euler[positive],marker=marker,linestyle='-',color=color,label=policy)
            if not np.all(positive):
                zero_times=np.array(times)[~positive]
                underflow.append(policy+': '+', '.join(f'{t:g}' for t in zero_times)+' s')
            trapezoidal=np.array([np.log(.5)]+[w['model_log_posterior'][1] for w in windows])
            positive_trapezoidal=trapezoidal<0
            axs[row,2].plot(np.array(times)[positive_trapezoidal],trapezoidal[positive_trapezoidal],
                marker=marker,linestyle='-',color=color,label=policy)
            if not np.all(positive_trapezoidal):
                trapezoidal_underflow.append(policy)
            axs[row,3].plot(ends,[len(w['selected']) for w in windows],marker=marker,linestyle='-',color=color,label=policy)
        axs[row,0].set(title=backend.upper()+' : Euler and Trapezoidal posteriors',
            ylim=(-.03,1.03),xlabel='Window-end time t (seconds)',ylabel='Model posterior probability (dimensionless)')
        axs[row,1].set_yscale('log')
        axs[row,1].set(title=backend.upper()+' : Euler model posterior (log scale)',
            xlabel='Window-end time t (seconds)',ylabel='Model posterior probability (log scale; dimensionless)')
        axs[row,3].set(title=backend.upper()+' : observations used',
            xlabel='Window-end time t (seconds)',ylabel='Observations assimilated per window (count)')
        axs[row,2].set_yscale('symlog',linthresh=1e-75)
        from matplotlib.ticker import FuncFormatter
        if backend=='laplace':
            axs[row,2].set_yticks([-10.**(-k) for k in range(0,71,10)])
            axs[row,2].yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:.0e}'))
        else:
            axs[row,2].set_yticks([-.1,-.2,-.3,-.4,-.5,-.6,-.7])
            axs[row,2].yaxis.set_major_formatter(FuncFormatter(lambda v,pos: f'{v:.1f}'))
        axs[row,2].set(title=backend.upper()+' : Trapezoidal log model posterior',
            xlabel='Window-end time t (seconds)',ylabel='ln(Model posterior probability): Trapezoidal\n(symlog display of negative log values)')
        if trapezoidal_underflow:
            axs[row,2].text(.02,.02,'Log posterior stored as zero / possible underflow; omitted: '+', '.join(trapezoidal_underflow),
                transform=axs[row,2].transAxes,fontsize=7,va='bottom')
        if underflow:
            axs[row,1].text(.02,.02,'Stored zero / possible underflow; omitted from log plot:\n'+'\n'.join(underflow),
                transform=axs[row,1].transAxes,fontsize=7,va='bottom')
        for ax in axs[row]:
            ax.grid(True,alpha=.25)
            if ax.lines: ax.legend(fontsize=8,loc='best',framealpha=.95)
            else: ax.text(.5,.5,'No completed arms; see results.json',ha='center',transform=ax.transAxes)
    fig.suptitle('Spring-mass model comparison | Initial model probabilities: 0.5 each at t = 0 s\n'
                 'Trapezoidal = explicit trapezoidal; lines connect window-end updates\n'
                 'Trapezoidal panel directly plots stable ln(P_T) from raw model scores, not a gap; linear float probabilities may round to 1.',fontsize=13)
    fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='spring_results')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--particles',type=int,default=64)
    parser.add_argument('--evidence-samples',type=int,default=1024)
    parser.add_argument('--duration',type=float,default=15.)
    parser.add_argument('--extra-points',type=int,default=4)
    parser.add_argument('--abc-eps-floor',type=float,default=.1)
    parser.add_argument('--abc-tolerance-policy',choices=['balanced','pooled'],default='balanced')
    parser.add_argument('--known-noise',action='store_true')
    args=parser.parse_args()
    cfg=Config(seed=args.seed,particles=args.particles,evidence_samples=args.evidence_samples,
               duration=args.duration,extra_points=args.extra_points,known_noise_schedule=args.known_noise,abc_eps_floor=args.abc_eps_floor,abc_tolerance_policy=args.abc_tolerance_policy)
    if cfg.particles<2 or cfg.evidence_samples<2 or cfg.duration<=0 or cfg.extra_points<0:
        parser.error('Need particles/evidence_samples >=2, duration>0, extra_points>=0')
    result=compare(cfg); save(result,args.output)
    for name,arm in result['arms'].items():
        print(name,arm.get('model_posterior',arm.get('reason')))
    print('Saved to',Path(args.output).resolve())

if __name__=='__main__': main()
