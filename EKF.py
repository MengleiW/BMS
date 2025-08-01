

import numpy as np
import scipy
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from numpy.linalg import inv
from scipy.stats import norm
def damped_oscillator(t, y, gamma, k):
    """
    Models a damped oscillator system.

    Inputs:
          t: Float, the current time point.
          y: List, current state of the system.
          gamma: Float, damping coefficient.
          k: Float, stiffness coefficient.

    Outputs:
         [y[1], f_t - gamma*y[1] - k*y[0]]: List, Derivative of the system state.
    """
    return [y[1], -gamma*y[1] - k*y[0]]

def Solution(t, gamma, k, y0):
    """
    Simulates the damped oscillator system over a given time span using solve_ivp.

    Inputs:
      gamma: Float, damping coefficient.
      k: Float, stiffness coefficient.
      initial_conditions: List, initial state of the system.
      ts: List, time span for simulation .
      N: Integer, number of points to evaluate in the time span.
      smooth_noise_level: number, level of random noise in the first half (smooth phase).
      jumpy_noise_level: number, level of random noise in the second half (jumpy phase).
    Outputs:
      (y, yp): list, where y is the position array and yp is the velocity array over the time span.
    """
    # Handle both scalar and array inputs for t
    if np.isscalar(t):
        t_eval = [t]
        return_scalar = True
    else:
        t_eval = [t] if isinstance(t, (int, float)) else t
        return_scalar = False
    
    # Solve the ODE using solve_ivp
    sol = scipy.integrate.solve_ivp(damped_oscillator, [0, max(t_eval) + 0.1], y0,  args=(gamma, k),  t_eval=t_eval)
    
    # Extract position and velocity
    position = sol.y[0]
    velocity = sol.y[1]
    
    if return_scalar:
        return np.array([position[0], velocity[0]])
    else:
        return np.array([position, velocity])

def euler_series(gamma, k, y0, t_grid):
    """
    Forward‑Euler integration from t_grid[0] to t_grid[-1]
    using time_steps internal steps per measurement interval.
    Returns 2×N array evaluated exactly at t_grid.
    """
    N   = len(t_grid)
    y   = np.zeros((2, N))
    y[:, 0] = y0
    time_steps=1
    
    
    for j in range(N - 1):
        # measurement interval
        h_big = t_grid[j + 1] - t_grid[j]
        h     = h_big / time_steps
        y_tmp = y[:, j].copy()

        for _ in range(time_steps):
            dy = damped_oscillator(None, y_tmp, gamma, k)
            y_tmp += h * np.asarray(dy)

        y[:, j + 1] = y_tmp

        if np.any(np.isnan(y_tmp)) or np.any(np.abs(y_tmp) > 1e6):
            return None                           # numerical blow‑up

    return y    

def trapezoidal_series(gamma, k, y0, t_grid):
    """
    Explicit trapezoidal scheme with the same sub‑stepping strategy.
    """
    N   = len(t_grid)
    y   = np.zeros((2, N))
    y[:, 0] = y0
    time_steps=1
    for j in range(N - 1):
        h_big = t_grid[j + 1] - t_grid[j]
        h     = h_big / time_steps
        y_tmp = y[:, j].copy()

        for _ in range(time_steps):
            f_n   = damped_oscillator(None, y_tmp, gamma, k)
            y_pred = y_tmp + h * np.asarray(f_n)
            f_np1  = damped_oscillator(None, y_pred, gamma, k)
            y_tmp += 0.5 * h * (np.asarray(f_n) + np.asarray(f_np1))

        y[:, j + 1] = y_tmp

        if np.any(np.isnan(y_tmp)) or np.any(np.abs(y_tmp) > 1e6):
            return None

    return y

def compute_log_likelihood(gamma, method, k, t_window,
                           y_obs, yp_obs, noise_level,
                           bounds=(0.01, 0.99)):
    """Gaussian log‑likelihood using BOTH position and velocity."""
    # parameter bounds
    if not (bounds[0] < gamma < bounds[1]):
        return -1e6

    # initial state taken from the first noisy sample in the window
    y0 = np.array([y_obs[0], yp_obs[0]])

    # forward model
    if method == 'euler':
        sol = euler_series(gamma, k, y0, t_window)
    elif method == 'trapezoidal':
        sol = trapezoidal_series(gamma, k, y0, t_window)
    else:                         # unsupported method
        return -1e6
    if sol is None:               # numerical blow‑up
        return -1e6

    y_pred, yp_pred = sol
    res_y  = y_obs  - y_pred
    res_yp = yp_obs - yp_pred


    log_lik_y  = -0.5 * len(y_obs) * np.log(2 * np.pi * noise_level ** 2) - 0.5 * np.sum(res_y  ** 2) / noise_level ** 2
    log_lik_yp = -0.5 *  len(y_obs) * np.log(2 * np.pi * noise_level ** 2) - 0.5 * np.sum(res_yp ** 2) / noise_level ** 2
    total_ll   = log_lik_y + log_lik_yp

    return np.clip(total_ll, -1e6, 1e6) if np.isfinite(total_ll) else -1e6

def estimate_gamma_map(method, k, t_window, y_obs, yp_obs, noise_level, bounds=(0.01, 0.99), prior_center=None, fisher_prior=0.0):
    """Estimate MAP (if prior_center provided) or MLE otherwise."""
    def log_post(g):
        log_lik = compute_log_likelihood(g, method, k, t_window, y_obs, yp_obs, noise_level)
        if prior_center is not None:
            log_prior = -0.5 * fisher_prior * (g - prior_center)**2
            return log_lik + log_prior
        return log_lik

    gamma_grid = np.linspace(bounds[0] + 0.01, bounds[1] - 0.01, 500)
    best_gamma = gamma_grid[0]
    best_val = -np.inf

    for g in gamma_grid:
        val = log_post(g)
        if val > best_val:
            best_val = val
            best_gamma = g

    res = minimize_scalar(lambda g: -log_post(g), bounds=bounds, method='bounded')
    if res.success and -res.fun > -best_val:
        best_gamma = res.x
        best_val = -res.fun

    return best_gamma, log_post(best_gamma)

def compute_model_evidence(method, k, t_window, y_obs, yp_obs, noise_level,
                           bounds=(0.01, 0.99), prior_center=None, fisher_prior=0.0):
    """
    Estimate MAP and compute model evidence using Laplace approximation.
    """
    gamma_map, log_post = estimate_gamma_map(
        method, k, t_window, y_obs, yp_obs, noise_level, bounds, prior_center, fisher_prior
    )

    def neg_log_post(g):
        return -compute_log_likelihood(g, method, k, t_window, y_obs, yp_obs, noise_level) - (
            0.5 * fisher_prior * (g - prior_center)**2 if prior_center is not None else 0.0)

    try:
        hess = scipy.optimize.approx_fprime(
            [gamma_map], lambda g: scipy.optimize.approx_fprime(
                g, neg_log_post, epsilon=1e-5
            )[0], epsilon=1e-5
        )[0]
    except:
        hess = 1e3

    fisher_post = max(hess, 1e-6)
    Z = np.exp(log_post) * np.sqrt(2 * np.pi / fisher_post)

    return gamma_map, Z, fisher_post




def build_joint_eps_schedule(k, t_window, y_obs, yp_obs, bounds=(0.01, 0.99), 
                            num_samples=200, quantiles=[80, 60, 40]):
    """
    Build a joint epsilon schedule for both models following Toni et al.
    This ensures fair model comparison by using the same tolerance for both.
    """
    rng = np.random.default_rng()
    y0 = np.array([y_obs[0], yp_obs[0]])
    obs_vec = np.vstack([y_obs, yp_obs])
    
    # Collect distances from both models
    all_distances = []
    
    for method in ['euler', 'trapezoidal']:
        distances = []
        for _ in range(num_samples // 2):  # Split samples between models
            gamma = rng.uniform(bounds[0] + 0.05, bounds[1] - 0.05, size=1)[0]
            
            if method == 'euler':
                sim = euler_series(gamma, k, y0, t_window)
            elif method == 'trapezoidal':
                sim = trapezoidal_series(gamma, k, y0, t_window)
            #elif method == 'analytical':
                #t_rel = t_window - t_window[0]
                #sol = np.asarray([Solution(t, gamma, k, y0) for t in t_rel])
                #sim = sol.T
            else:
                raise ValueError("unknown method")
            
            
            if sim is not None:
                dist = np.linalg.norm(obs_vec - sim)
                if dist < np.inf:
                    distances.append(dist)
        all_distances.extend(distances)
    
    # Compute quantiles from combined distances
    if len(all_distances) < 10:
        return np.array([1.0, 0.5, 0.2])  # Fallback
    
    eps_schedule = np.percentile(all_distances, quantiles)
    
    # Ensure minimum epsilon to avoid being too strict
    eps_schedule = np.maximum(eps_schedule, 0.1)
    
    return eps_schedule

def abc_smc_estimation(method, k, t_window, y_obs, yp_obs,
                       bounds=(0.01, 0.99), eps_schedule=None,
                       num_particles=200, num_generations=3,
                       max_total_attempts=150_000,
                       min_accept_rate=5e-4,
                       verbose=False,
                       prior_particles=None, prior_weights=None):
    """
    Adaptive ABC–SMC for the damping γ parameter.
    Returns: gamma_mean, evidence_proxy, particles_last, weights_last
    """
    rng = np.random.default_rng()
    particles = np.zeros((num_generations, num_particles))
    weights = np.zeros_like(particles)
    
    y0 = np.array([y_obs[0], yp_obs[0]])
    obs_vec = np.vstack([y_obs, yp_obs])

    if eps_schedule is None:
        eps_schedule = build_joint_eps_schedule(k, t_window, y_obs, yp_obs, bounds)
    
    total_attempts = 0
    actual_particles = num_particles
    prev_actual_particles = num_particles

    for t in range(num_generations):
        eps_t = eps_schedule[min(t, len(eps_schedule) - 1)]
        if verbose:
            print(f"[ABC-{method}] Gen {t+1}/{num_generations}, ε = {eps_t:.3f}")
        
        acc = 0
        attempts = 0
        temp_particles, temp_distances = [], []

        if t > 0:
            prev_particles = particles[t-1, :prev_actual_particles]
            prev_weights = weights[t-1, :prev_actual_particles]

            weight_sum = np.sum(prev_weights)
            if weight_sum <= 1e-12:
                if verbose:
                    print(f"[ABC-{method}] WARNING: degenerate weights in generation {t}, falling back to uniform.")
                prev_weights = np.ones_like(prev_particles) / len(prev_particles)
            else:
                prev_weights = prev_weights / weight_sum

            weighted_mean = np.average(prev_particles, weights=prev_weights)
            weighted_var = np.average((prev_particles - weighted_mean) ** 2, weights=prev_weights)
            sigma = max(0.01, 0.10 * np.sqrt(weighted_var))

        while acc < num_particles and total_attempts < max_total_attempts:
            attempts += 1
            total_attempts += 1

            if t == 0:
                if prior_particles is not None and prior_weights is not None:
                    g_proposed = rng.choice(prior_particles, p=prior_weights)
                else:
                    g_proposed = rng.uniform(bounds[0] + 0.05, bounds[1] - 0.05)
            else:
                idx = rng.choice(prev_actual_particles, p=prev_weights)
                g_center = prev_particles[idx]
                g_proposed = np.clip(g_center + rng.normal(0, sigma),
                                     bounds[0] + 0.05, bounds[1] - 0.05)

            # Simulate
            if method == 'euler':
                sim = euler_series(g_proposed, k, y0, t_window)
            elif method == 'trapezoidal':
                sim = trapezoidal_series(g_proposed, k, y0, t_window)
            else:
                raise ValueError("Unknown method")

            if sim is None:
                continue

            dist = np.linalg.norm(obs_vec - sim)
            if dist <= eps_t:
                temp_particles.append(g_proposed)
                temp_distances.append(dist)
                acc += 1

        if acc == 0:
            if verbose:
                print(f"[ABC-{method}] ERROR: No particles accepted in generation {t}.")
            return 0.3, 1e-10, np.array([0.3]), np.array([1.0])

        # Save accepted particles
        actual_particles = acc
        for i in range(acc):
            particles[t, i] = temp_particles[i]

        if t == 0:
            weights[t, :acc] = 1.0 / acc
        else:
            temp_weights = np.zeros(acc)
            for j in range(acc):
                kernel_vals = np.exp(-0.5 * ((particles[t, j] - prev_particles) / sigma) ** 2)
                denom = np.sum(prev_weights * kernel_vals)
                temp_weights[j] = 1.0 / max(denom, 1e-12)

            weight_sum = np.sum(temp_weights)
            if weight_sum > 0:
                weights[t, :acc] = temp_weights / weight_sum
            else:
                weights[t, :acc] = np.ones(acc) / acc

        prev_actual_particles = acc

        if verbose:
            ar = acc / attempts
            print(f"[ABC-{method}] Accepted {acc}/{attempts} (rate: {ar:.3g})")

    # Final posterior summary
    final_particles = particles[num_generations - 1, :actual_particles]
    final_weights = weights[num_generations - 1, :actual_particles]

    weight_sum = np.sum(final_weights)
    if weight_sum <= 1e-12:
        if verbose:
            print(f"[ABC-{method}] WARNING: Final weights degenerate. Using uniform fallback.")
        final_weights = np.ones_like(final_particles) / len(final_particles)
    else:
        final_weights /= weight_sum

    g_mean = np.average(final_particles, weights=final_weights)
    g_std = np.sqrt(np.average((final_particles - g_mean) ** 2, weights=final_weights))
    evidence = max((actual_particles / max(total_attempts, 1)) / eps_schedule[-1], 1e-12)

    if verbose:
        print(f"[ABC-{method}] Done. γ = {g_mean:.3f} ± {g_std:.3f}, evidence ≈ {evidence:.3e}")

    return g_mean, evidence, final_particles, final_weights

def extended_kalman_filter(y_obs, yp_obs, gamma_ekf, k, dt=0.5, R_scale=0.01, Q_scale=0.01):
    """
    Extended Kalman Filter for the damped harmonic oscillator using both position and velocity observations.

    Parameters
    ----------
    y_obs : ndarray, shape (T,)
        Observed positions.
    yp_obs : ndarray, shape (T,)
        Observed velocities.
    gamma_ekf : float
        Model-averaged damping coefficient.
    k : float
        Spring constant.
    dt : float
        Time step.
    R_scale : float
        Measurement noise variance scale (applied to both pos & vel).
    Q_scale : float
        Process noise variance scale (for state dynamics).

    Returns
    -------
    x_filtered : ndarray, shape (2, T)
        Filtered states [position; velocity] over time.
    P_filtered : ndarray, shape (2, 2, T)
        State covariance at each time.
    """
    T = len(y_obs)
    x_filtered = np.zeros((2, T))
    P_filtered = np.zeros((2, 2, T))

    # Initial estimate (position from y_obs, assume velocity = 0)
    x = np.array([y_obs[0], yp_obs[0]])
    P = np.eye(2) * 0.1

    A = np.array([[1, dt],
                  [-k * dt, 1 - gamma_ekf * dt]])
    H = np.eye(2)  # Measure both position and velocity
    R = np.eye(2) * R_scale
    Q = np.eye(2) * Q_scale

    for t in range(T):
        # Prediction
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # Observation
        y_meas = np.array([y_obs[t], yp_obs[t]])
        y_pred = H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        if np.linalg.norm(K) < 1e-5:
            print("Warning: Kalman gain ~0 → no update in EKF!")
        # Update
        x = x_pred + K @ (y_meas - y_pred)
        P = (np.eye(2) - K @ H) @ P_pred

        x_filtered[:, t] = x           #new states
        P_filtered[:, :, t] = P      #State covariance

    return x_filtered, P_filtered
def adapt_window_size(observed_y, predicted_y, current_window_size,
                      W_min=4, W_max=20, delta_s=2, epsilon_r_prev=None, lambda_penalty=0.2):
    """
    Adapt the time window size based on the penalized NRMSD between observed and predicted values.

    Parameters
    ----------
    observed_y : array-like
        Observed values for the current window.
    predicted_y : array-like
        Model-predicted values for the current window.
    current_window_size : int
        Current time window size (in time steps).
    W_min : int
        Minimum allowed window size.
    W_max : int
        Maximum allowed window size.
    delta_s : int
        Amount by which to increase or decrease window size.
    epsilon_r_prev : float or None
        Previous penalized NRMSD (for comparison).
    lambda_penalty : float
        Weight for window size penalty (recommended: 0.1–0.5)

    Returns
    -------
    new_window_size : int
        Updated window size.
    new_epsilon_r : float
        Penalized NRMSD.
    """
    observed_y = np.asarray(observed_y)
    predicted_y = np.asarray(predicted_y)

    if len(observed_y) != len(predicted_y):
        raise ValueError("Observed and predicted arrays must be the same length.")

    # Compute raw NRMSD
    error = observed_y - predicted_y
    rmse = np.sqrt(np.mean(error ** 2))
    y_range = np.max(observed_y) - np.min(observed_y)
    epsilon_r = rmse / y_range if y_range != 0 else 0.0

    # Apply penalty for longer windows
    penalty = lambda_penalty * (current_window_size / W_max)
    epsilon_r_adj = epsilon_r + penalty

    # Adapt window size based on penalized error
    if epsilon_r_prev is None:
        new_window_size = current_window_size
    elif epsilon_r_adj > epsilon_r_prev:
        new_window_size = max(W_min, current_window_size - delta_s)
    else:
        new_window_size = min(W_max, current_window_size + delta_s)

    return new_window_size, epsilon_r_adj

def direct_marginal_likelihood(gamma_hat, k, y0, t_window, y_obs, yp_obs, noise_std, bounds=(0.01, 0.99)):
    """
    Compute marginal likelihood Z = ∫ p(y | γ) p(γ) dγ using exact solution and uniform prior.
    No inner function or particles — pure direct integration.
    """
    gamma_grid = np.linspace(bounds[0], bounds[1], 100)

    total_Z = 0.0
    for i in range(len(gamma_grid) - 1):
        gamma_left = gamma_grid[i]
        gamma_right = gamma_grid[i+1]
        gamma_mid = 0.5 * (gamma_left + gamma_right)

        try:
            sol = Solution(t_window, gamma_mid, k, y0)
            y_pred = sol[0, :]
            yp_pred = sol[1, :]
        except Exception:
            continue

        log_p_y = norm.logpdf(y_obs, loc=y_pred, scale=noise_std)
        log_p_yp = norm.logpdf(yp_obs, loc=yp_pred, scale=noise_std)
        log_likelihood = np.sum(log_p_y + log_p_yp)

        # Uniform prior density
        log_prior = -np.log(bounds[1] - bounds[0])
        log_weight = log_likelihood + log_prior

        Z_piece = np.exp(log_weight) * (gamma_right - gamma_left)
        total_Z += Z_piece

    return total_Z


def run_simple_analysis():
    """Run the simplified damped oscillator analysis"""
    
    # Parameters
    k = 1.0  # Spring constant
    gamma_true = 0.3  # True damping coefficient
    y0_true = [1.0, 0.0]  # Initial conditions [position, velocity]
    dt = 0.5  # Time step (0.5s per observation)
    T_total = 15  # Total time (15 seconds)
    noise_level = 0.1  # Observation noise
    window_size_seconds = 3 # Window size in seconds
    W_min = 4
    W_max = 20
    delta_s = 2
    # Generate time array
    t = np.arange(0, T_total + dt, dt)
    n_points = len(t)
    
    # Generate true solution
    true_sol = Solution(t, gamma_true, k, y0_true)
    true_y = true_sol[0, :]
    true_yp = true_sol[1, :]
    
    # Add noise to observations
    np.random.seed(42)
    observed_y = true_y + np.random.normal(0, noise_level, n_points)
    observed_yp = true_yp + np.random.normal(0, noise_level, n_points)
    
    # Window analysis
    window_size = int(window_size_seconds / dt)  # 8 observations per window
    n_windows = int((n_points - 1) / window_size)
    
    # Storage for results
    window_centers = []
    euler_gammas = []
    trap_gammas = []
    euler_evidence = []
    trap_evidence = []
    abc_gamma_euler = []
    abc_gamma_trap = []
    abc_evidence_eu = []
    abc_evidence_tr = []
    direct_evidence=[]
    direct_evidence_full = np.zeros_like(t) 
    prior_center_euler, fisher_euler = None, 0.0
    prior_center_trap, fisher_trap = None, 0.0
    prev_particles_euler, prev_weights_euler = None, None
    prev_particles_trap, prev_weights_trap = None, None
    x0_ekf_next = None
    epsilon_r_prev = None
    
    
    # Storage for predictions (for plotting)
    euler_predictions = np.zeros_like(true_y)
    trap_predictions = np.zeros_like(true_y)
    abc_euler_predictions = np.zeros_like(true_y)
    abc_trap_predictions = np.zeros_like(true_y)
    
    print(f"Running simplified analysis:")
    print(f"  Total time: {T_total}s")
    print(f"  Time step: {dt}s")
    print(f"  Window size: {window_size_seconds}s ({window_size} observations)")
    print(f"  Number of windows: {n_windows}")
    print(f"  True gamma: {gamma_true}")
    print(f"  Noise level: {noise_level}")
    print()
    
    # Process each window
    i = 0
    start_idx = 0
    epsilon_r_prev = None
    current_window_size = window_size  # initialize
    window_bounds = []
    
    
    
    while start_idx + 2 <= len(t):
        if start_idx + current_window_size + 1 > len(t):
            current_window_size = len(t) - start_idx - 1
        end_idx = start_idx + current_window_size + 1
        window_bounds.append((start_idx, end_idx))
        
       
       
    
        # Extract window data
        t_window = t[start_idx:end_idx]
        y_window = observed_y[start_idx:end_idx]
        yp_window = observed_yp[start_idx:end_idx]
        t_window_rel = t_window - t_window[0]
    
        window_center = np.mean(t_window)
        window_centers.append(window_center)
    
        print(f"Window {i+1}: t = {t[start_idx]:.1f}s to {t[start_idx] + current_window_size * dt:.1f}s (W = {current_window_size})")
    
            
        # Estimate parameters for each method
        # Euler method
        gamma_euler, evidence_euler_i, fisher_euler = compute_model_evidence(
            'euler', k, t_window_rel, y_window, yp_window, noise_level,
            prior_center=prior_center_euler, fisher_prior=fisher_euler
        )
        prior_center_euler = gamma_euler
        euler_gammas.append(gamma_euler)
        euler_evidence.append(evidence_euler_i)
        print(f"  Euler: γ = {gamma_euler:.3f} (error = {abs(gamma_euler - gamma_true):.3f})")
        
        # Trapezoidal method
        gamma_trap, evidence_trap_i, fisher_trap = compute_model_evidence(
            'trapezoidal', k, t_window_rel, y_window, yp_window, noise_level,
            prior_center=prior_center_trap, fisher_prior=fisher_trap
        )
        prior_center_trap = gamma_trap
        trap_gammas.append(gamma_trap)
        trap_evidence.append(evidence_trap_i)
        print(f"  Trapezoidal: γ = {gamma_trap:.3f} (error = {abs(gamma_trap - gamma_true):.3f})")
        
        # Generate predictions for this window
        if i == 0:
            y0_window = np.array([y_window[0], yp_window[0]])
        else:
            y0_window = x0_ekf_next
        
        # Euler prediction
        euler_sol = euler_series(gamma_euler, k, y0_window, t_window_rel)
        if euler_sol is not None:
            euler_predictions[start_idx:end_idx] = euler_sol[0, :]
        
        # Trapezoidal prediction
        trap_sol = trapezoidal_series(gamma_trap, k, y0_window, t_window_rel)
        if trap_sol is not None:
            trap_predictions[start_idx:end_idx] = trap_sol[0, :]
        
        # ABC-SMC with JOINT epsilon schedule (following Toni et al.)
        print("  Building joint epsilon schedule...")
        eps_schedule = build_joint_eps_schedule(
            k, t_window_rel, y_window, yp_window,
            bounds=(0.01, 0.99),
            num_samples=200,
            quantiles=[50, 30, 20]  # Three-stage ladder
        )
        print(f"  Joint ε schedule: {eps_schedule}")
        
        # Run Euler with joint schedule
        # Euler ABC
        res_eu = abc_smc_estimation(
            'euler', k, t_window_rel, y_window, yp_window,
            eps_schedule=eps_schedule,
            verbose=False,
            prior_particles=prev_particles_euler,
            prior_weights=prev_weights_euler
        )
        g_eu_abc, Z_eu_abc, final_particles_eu, final_weights_eu = res_eu

        
        # Update prior for next window
        prev_particles_euler = final_particles_eu
        final_particles_eu = np.atleast_1d(final_particles_eu)
        prev_weights_euler = np.ones_like(final_particles_eu) / len(final_particles_eu)


        
        # Run Trapezoidal with SAME schedule
        # Trapezoidal ABC
        res_tr = abc_smc_estimation(
            'trapezoidal', k, t_window_rel, y_window, yp_window,
            eps_schedule=eps_schedule,
            verbose=False,
            prior_particles=prev_particles_trap,
            prior_weights=prev_weights_trap
        )
        g_tr_abc,  Z_tr_abc, final_particles_tr, final_weights_tr = res_tr
        
        
        # Update prior for next window
        prev_particles_trap = final_particles_tr
    
        final_particles_tr = np.atleast_1d(final_particles_tr)
        prev_weights_trap = np.ones_like(final_particles_tr) / len(final_particles_tr)

        
        print(f"  ABC-Euler: γ = {g_eu_abc:.3f}, evidence = {Z_eu_abc:.3e}")
        print(f"  ABC-Trap: γ = {g_tr_abc:.3f}, evidence = {Z_tr_abc:.3e}")
        
        # ABC trajectory predictions
        sol_eu_abc = euler_series(g_eu_abc, k, y0_window, t_window_rel)
        if sol_eu_abc is not None:
            abc_euler_predictions[start_idx:end_idx] = sol_eu_abc[0, :]
        
        sol_tr_abc = trapezoidal_series(g_tr_abc, k, y0_window, t_window_rel)
        if sol_tr_abc is not None:
            abc_trap_predictions[start_idx:end_idx] = sol_tr_abc[0, :]
        
        abc_gamma_euler.append(g_eu_abc)
        abc_gamma_trap.append(g_tr_abc)
        abc_evidence_eu.append(Z_eu_abc)
        abc_evidence_tr.append(Z_tr_abc)
        
        
        #EKF FUNTION
        
        gamma_map = (gamma_euler + gamma_trap) / 2
        gamma_abc = (g_eu_abc + g_tr_abc) / 2
        Z_map = (evidence_euler_i + evidence_trap_i) / 2
        Z_abc = (Z_eu_abc + Z_tr_abc) / 2
        gamma_ekf = (Z_map * gamma_map + Z_abc * gamma_abc) / (2 * (Z_map + Z_abc + 1e-12))

        gamma_ekf = (Z_map * gamma_map + Z_abc * gamma_abc) / (2 * (Z_map + Z_abc + 1e-12))

        x_filtered, P_filtered = extended_kalman_filter(y_window, yp_window, gamma_ekf, k, dt=dt)

        x0_ekf_next = x_filtered[:, -1]  # store for next window
        
        Z_direct = direct_marginal_likelihood(
            gamma_hat=gamma_ekf,
            k=k,
            y0=y0_window,
            t_window=t_window_rel,
            y_obs=y_window,
            yp_obs=yp_window,
            noise_std=noise_level
        )
        print(f"  Direct Evidence (uniform prior): {Z_direct:.3e}")
        direct_evidence.append(Z_direct)
        
        
        #time window
        # Step: Adaptive window update
        
        residuals = []
        
        # Euler prediction residual
        if sol_eu_abc is not None:
            residuals.append(np.abs(y_window - sol_eu_abc[0, :]))
        
        # Trapezoidal prediction residual
        if sol_tr_abc is not None:
            residuals.append(np.abs(y_window - sol_tr_abc[0, :]))
        
        # Average residual
        if residuals:
            avg_residual = np.mean(residuals, axis=0)
            predicted_avg = y_window - avg_residual  # crude "denoised" estimate
        else:
            predicted_avg = x_filtered[0, :]
        
        new_window_size, epsilon_r_prev = adapt_window_size(
            observed_y=y_window,
            predicted_y=predicted_avg,
            current_window_size=window_size,
            W_min=W_min,
            W_max=W_max,
            delta_s=delta_s,
            epsilon_r_prev=epsilon_r_prev
        )

        window_size = new_window_size
        current_window_size = new_window_size
        start_idx = end_idx - 1
        print(f"[Adapt] ε_r = {epsilon_r_prev:.4f}, new window size = {window_size}")

        i += 1
                
        

    bayes_factors = [trap_evidence[i] / euler_evidence[i] if euler_evidence[i] > 0 else 1 
                     for i in range(len(euler_evidence))]

    # Build full-length evidence & Bayes-factor signals
    evidence_euler_full = np.zeros_like(t)
    evidence_trap_full = np.zeros_like(t)
    bayes_full = np.zeros_like(t)
    
    
    for i, (s, e) in enumerate(window_bounds):
        direct_evidence_full[s:e] = direct_evidence[i]
        
        
    for i in range(len(euler_evidence)):
        s = sum([window_size_seconds] * i) // dt  # cumulative start index
        e = s + window_size + 1
        if e > len(t):  # prevent index error
            e = len(t)
        s = int(s)
        e = int(e)
        evidence_euler_full[s:e] = euler_evidence[i]
        evidence_trap_full[s:e] = trap_evidence[i]
        bayes_full[s:e] = trap_evidence[i] / euler_evidence[i] if euler_evidence[i] > 0 else np.inf

    
    # Piece-wise-constant ABC evidence over the whole timeline
    abc_evidence_eu_full = np.zeros_like(t, dtype=float)
    abc_evidence_tr_full = np.zeros_like(t, dtype=float)
    
    for i, (s, e) in enumerate(window_bounds):
        abc_evidence_eu_full[s:e] = abc_evidence_eu[i]
        abc_evidence_tr_full[s:e] = abc_evidence_tr[i]

    
    # Create plots
    print(f"\nCreating plots...")
    
    # Plot 1 - Vanilla ABC trajectories
    plt.figure(figsize=(10, 6))
    plt.plot(t, true_y, 'g-', linewidth=2, label='True solution')
    plt.plot(t, abc_euler_predictions, 'b--', linewidth=2, label='Forward Euler (ABC)')
    plt.plot(t, abc_trap_predictions, 'r--', linewidth=2, label='Trapezoidal rule (ABC)')
    plt.plot(t, observed_y, 'ko', markersize=4, alpha=0.5, label='Measurements')
    
    for i in range(n_windows + 1):
        window_time = i * window_size_seconds
        if window_time <= T_total:
            plt.axvline(window_time, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.title('adptive ABC (with joint ε schedule)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Position vs Time (MLE)
    plt.figure(figsize=(10, 6))
    plt.plot(t, true_y, 'g-', linewidth=2, label='True solution')
    plt.plot(t, euler_predictions, 'b--', linewidth=2, label='Forward Euler')
    plt.plot(t, trap_predictions, 'r--', linewidth=2, label='Trapezoidal rule')
    plt.plot(t, observed_y, 'ko', markersize=4, alpha=0.5, label='Measurements')
    
    for i in range(n_windows + 1):
        window_time = i * window_size_seconds
        if window_time <= T_total:
            plt.axvline(window_time, color='gray', linestyle=':', alpha=0.5)
    
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Position', fontsize=12)
    plt.title('adptive Laplace', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 3 - MLE evidence & Bayes factor (with L2 norm evidence)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # (a) Evidence (MLE)
    ax1.step(t, evidence_euler_full, where='post', color='blue', label='Euler (Laplace)')
    ax1.step(t, evidence_trap_full, where='post', color='red', label='Trap (MLaplace)')
    ax1.step(t, direct_evidence_full, where='post', color='black', ls='-', label='Direct (Uniform Prior)')

    ax1.set_yscale('log')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Model evidence (1/L2 norm)')
    ax1.set_title('(a) Evidence — Laplace')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # (b) Bayes factor (MLE)
    bf_mle = evidence_trap_full / evidence_euler_full
    ax2.step(t, bf_mle, where='post', color='purple', label='Trap/Euler (Laplace)')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Bayes factor')
    ax2.set_title('(b) Bayes factor —Laplace')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.suptitle('Figure 3 —  Laplace: Evidence (L2 norm) & Bayes factor')
    plt.tight_layout()
    plt.show()
    
    # Figure 4 — ABC-SMC evidence & Bayes factor

    # Rebuild evidence traces with adaptive window spans
    t_piecewise = []
    abc_eu_piecewise = []
    abc_tr_piecewise = []
    direct_evidence_piecewise = []
    for i, (s, e) in enumerate(window_bounds):
        t_piecewise.extend(t[s:e])
        abc_eu_piecewise.extend([abc_evidence_eu[i]] * (e - s))
        abc_tr_piecewise.extend([abc_evidence_tr[i]] * (e - s))
        direct_evidence_piecewise.extend([direct_evidence[i]] * (e - s))
    # Compute piecewise Bayes factor
    bf_piecewise = [tr / eu if eu > 0 else np.inf for tr, eu in zip(abc_tr_piecewise, abc_eu_piecewise)]
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # (a) ABC-SMC Evidence
    ax1.step(t_piecewise, abc_eu_piecewise, where='post', color='cyan', ls='--', label='Euler (ABC)')
    ax1.step(t_piecewise, abc_tr_piecewise, where='post', color='orange', ls='--', label='Trap (ABC)')
    ax1.step(t_piecewise, direct_evidence_piecewise, where='post', color='black', ls='-', label='Direct (Uniform Prior)')


    ax1.set_yscale('log')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Model evidence')
    ax1.set_title('(a) Evidence — ABC-SMC (joint ε)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # (b) ABC-SMC Bayes factor
    ax2.step(t_piecewise, bf_piecewise, where='post', color='green', ls='--', label='Trap/Euler (ABC)')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Bayes factor')
    ax2.set_title('(b) Bayes factor — ABC-SMC')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.suptitle('Figure 4 — ABC-SMC: Evidence & Bayes factor (joint ε schedule)')
    plt.tight_layout()
    plt.show()

    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Euler mean γ error: {np.mean([abs(g - gamma_true) for g in euler_gammas]):.4f}")
    print(f"  Trapezoidal mean γ error: {np.mean([abs(g - gamma_true) for g in trap_gammas]):.4f}")
    print(f"  Average Bayes factor (Trap/Euler): {np.mean(bayes_factors):.2f}")

# Run the analysis
if __name__ == "__main__":
    run_simple_analysis()
