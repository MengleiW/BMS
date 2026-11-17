# Spring-mass window comparison: ratio selection and stable log posteriors

This standalone research example compares three window policies with shared Laplace and ABC-SMC inference. The included example output uses the unchanged default configuration.

## Attachment rule implemented
Incoming state, parameter estimates and model weights are frozen for full-window predictions. All available observations inform diagnostic MSE and selection only. Subsequent inference, model-weight updates and EKF observation corrections use selected observations only. From window 3 onward:

`extra_count_raw = previous_actual_extra_count * current_full_window_MSE / mean(previous_two_full_window_MSEs)`

The attachment's final proportional rule replaces its earlier doubling/halving proposal. First two windows target twice the baseline sampling density, subject to available observations. Extra points are allocated proportionally by segment MSE and distributed by filling uncovered gaps. Baseline points include the final observation and respect the covariance-based propagation restriction.

## Explicit implementation conventions, not mathematical claims
- MSE retains the existing mean of squared position and velocity residuals, equally weighted and without normalization. These components can have different physical units. This is a preserved code convention, not a scale rule supplied by the attachment; component MSEs are logged separately.
- The shared starting point initializes the state and is excluded from baseline/new observation counts. Twice baseline means baseline_new + baseline_new extras, capped by grid capacity.
- A greedy uncertainty pass estimates baseline count. That count is evenly distributed across the uniform observation-time grid, with nearest-grid rounding (half up), deduplication and endpoint retention. All actual gaps are then rechecked sequentially, including intermediate prediction uncertainties; unsafe gaps receive earlier available points. The full path is recalculated after each insertion. This heuristic is not a proof of minimal baseline count or globally optimal placement.
- Planning covariance uses conditional hypothetical corrections, not actual observation values; incoming state and parameter information remain unchanged. The existing maximum position standard deviation 0.2 is retained, checked for both models. The grid is declared infeasible if a one-step gap cannot satisfy it.
- Extra counts use nearest-integer half-up rounding, then remaining-capacity capping. Historical MSE uses exactly the previous two windows of the same backend, before their inference updates. No noise floor is introduced.
- If reference MSE is exactly zero and current MSE is positive, fill remaining capacity. If both are zero, retain previous extra count. If previous extras are zero and current MSE is positive, restart with at least one extra point (capacity permitting). These are explicit conventions filling unresolved attachment cases.
- The final shorter window inherits an absolute previous extra count, then applies the ratio and capacity cap; no hidden density rescaling.
- `extra_points` remains a legacy compatibility field but does not control the new ratio budget. Other inference/noise settings are unchanged (noise standard deviation 0.01 throughout).

## Run
Use Python 3.10+ with existing NumPy/SciPy/Matplotlib dependencies:

```bash
python -m unittest discover -v
python spring_window_comparison.py --output new_results
```

The old full-selection equivalence test now explicitly injects a full-capacity budget rather than relying on the obsolete `extra_points` setting. New tests cover warmup, historical reference, rounding, zero cases, capacity limits, baseline gaps and log consistency.

## Audit output
Each selected-window record contains baseline points, initial even-grid points, repair insertions, baseline/final gap checks, frozen full predictions, point/component MSE, full MSE, historical reference, ratio, previous/actual extra counts, allocation and indices. The original six-panel arrangement is retained: model probabilities, Euler log probability, observation counts, for Laplace and ABC.

This remains a heuristic, data-dependent sampling experiment. Diagnostics use all observations, selection bias is possible, ABC tolerances and approximation errors remain, and successful execution does not establish optimality, calibrated model probabilities or computational savings.


## Stable posterior representation (display-only diagnostic)
This copy preserves all original numeric inference and recursion. Before the existing model normalization, raw model log scores are converted to a stable binary log posterior with a sign-split log1p formula. New per-window fields `model_log_posterior`, `model_posterior_decimal`, and `model_log_odds_trapezoidal_over_euler` preserve/display near-one information without taking log of a rounded probability. Legacy numeric `model_posterior` remains for compatibility and may still equal 1.0 in float64. Decimal strings exponentiate the stored finite log probability at adaptive precision: they do not imply greater statistical accuracy or recovered precision in the input scores.

The third plot column directly displays ln(P_Trapezoidal), on a symlog display (linear threshold 1e-75). It is not 1-P and not log(1-P). Closer to zero means larger posterior probability. Underflow to a zero log probability is explicitly flagged/omitted; diagnostic log odds remain available. No inference algorithm, parameter, noise, seed or point-selection policy is changed. The original six plots remain, plus the two direct Trapezoidal log-posterior panels.


## Included example
`example_results/comparison.png` is an eight-panel comparison: each row is Laplace or ABC; columns are both model posterior probabilities, Euler log-scale probability, direct stable Trapezoidal log posterior, and assimilated-observation count. `example_results/results.json` contains the matching reproducible default run and detailed diagnostics. Run to a new directory to preserve the example. Internal identifier `Heun` denotes the explicit Trapezoidal scheme; figures use Trapezoidal consistently.

Run `python -m unittest discover -v` for the 22 existing inference/selection tests plus three stable-log representation tests. No authentication, network service, or private dataset is needed.
