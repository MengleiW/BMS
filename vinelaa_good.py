import numpy as np
import scipy
import scipy.stats
import scipy.integrate
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from numpy.linalg import inv

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

def estimate_gamma_mle(method, k, t_window, y_obs, yp_obs, noise_level, bounds=(0.01, 0.99), prior_center=None):
    """Estimate gamma using MLE for a given method"""
    
    # Grid search for initial guess
    if prior_center is not None:
        gamma_grid = np.linspace(max(bounds[0], prior_center - 0.5), min(bounds[1], prior_center + 0.5), 500)
    else:
        gamma_grid = np.linspace(bounds[0] + 0.05, bounds[1] - 0.05, 500)

    best_ll = -1e6
    best_gamma = 0.3
    
    for gamma_test in gamma_grid:
        ll = compute_log_likelihood(gamma_test, method, k, t_window, y_obs, yp_obs, noise_level)
        if ll > best_ll:
            best_ll = ll
            best_gamma = gamma_test
    
    # Refine with optimization
    try:
        result = minimize_scalar(
        lambda g: -compute_log_likelihood(g, method, k, t_window, y_obs, yp_obs, noise_level),
        bounds=bounds if prior_center is None else (
            max(bounds[0], prior_center - 0.5), min(bounds[1], prior_center + 0.5)),
        method='bounded'
    )

        if result.success and -result.fun > best_ll:
            best_gamma = result.x
            best_ll = -result.fun
    except:
        pass
    
    return best_gamma, best_ll



def compute_model_evidence(method, k, t_window, y_obs, yp_obs, noise_level,
                           bounds=(0.01, 0.99), prior_center=None):
    """
    Estimate MLE and compute model evidence using Laplace approximation.
    """
    # Find MLE
    gamma_mle, log_likelihood = estimate_gamma_mle(
        method, k, t_window, y_obs, yp_obs, noise_level, bounds, prior_center
    )
    
    # Compute Fisher information (negative 2nd derivative)
    def neg_log_likelihood(g):
        return -compute_log_likelihood(g, method, k, t_window, y_obs, yp_obs, noise_level)

    try:
        hess = scipy.optimize.approx_fprime(
            [gamma_mle], lambda g: scipy.optimize.approx_fprime(
                g, neg_log_likelihood, epsilon=1e-5
            )[0], epsilon=1e-5
        )[0]
    except:
        hess = 1e3  # fallback high curvature if numerical differentiation fails

    fisher_info = max(hess, 1e-6)  # avoid division by zero or negative
    
    # Laplace approximation of marginal likelihood
    prior_density_at_mle =1/2
    Z = np.exp(log_likelihood) *  prior_density_at_mle * np.sqrt(2 * np.pi / fisher_info)

    return gamma_mle, Z



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
                       prev_particles=None, prev_weights=None):

    """
    Adaptive ABC–SMC for the damping γ parameter.
    Modified to use external epsilon schedule if provided.
    
    Returns
    -------
    gamma_mean, gamma_std, evidence_proxy, particles_last
    """
    rng = np.random.default_rng()
    particles = np.zeros((num_generations, num_particles))
    weights = np.zeros_like(particles)
    distances = np.zeros(num_particles)
    
    y0 = np.array([y_obs[0], yp_obs[0]])
    obs_vec = np.vstack([y_obs, yp_obs])
    
    # Use provided epsilon schedule or compute one
    if eps_schedule is None:
        # Build schedule if not provided
        eps_schedule = build_joint_eps_schedule(k, t_window, y_obs, yp_obs, bounds)
    
    total_attempts = 0
    actual_particles = num_particles  # Track actual number of particles
    prev_actual_particles = num_particles  # Track from previous generation
    
    for t in range(num_generations):
        eps_t = eps_schedule[min(t, len(eps_schedule)-1)]
        
        if verbose:
            print(f"[ABC-{method}] Gen {t+1}/{num_generations}, ε = {eps_t:.3f}")
        
        acc = 0
        attempts = 0
        temp_particles = []
        temp_distances = []
        
        # Calculate proposal parameters before sampling
        if t > 0:
            # Use actual number of particles from previous generation
            prev_particles = particles[t-1, :prev_actual_particles]
            prev_weights = weights[t-1, :prev_actual_particles] 
            prev_weights = prev_weights / np.sum(prev_weights)  # Renormalize
            
            # Global adaptive sigma
            weighted_mean = np.average(prev_particles, weights=prev_weights)
            weighted_var = np.average((prev_particles - weighted_mean)**2, weights=prev_weights)
            sigma = 0.10 * np.sqrt(weighted_var)
            sigma = max(sigma, 0.01)  # Minimum sigma
        
        while acc < num_particles and total_attempts < max_total_attempts:
            attempts += 1
            total_attempts += 1
            
            # Propose gamma
            if t == 0:
                g_proposed = rng.uniform(bounds[0] + 0.05, bounds[1] - 0.05, size=1)[0]
            else:
                # Multinomial resampling from actual particles
                idx = rng.choice(prev_actual_particles, p=prev_weights)
                g_center = prev_particles[idx]
                g_proposed = np.clip(g_center + rng.normal(0, sigma),
                                   bounds[0] + 0.05, bounds[1] - 0.05)
            
            # Simulate and compute distance
            
            if method == 'euler':
                sim = euler_series(g_proposed, k, y0, t_window)
            elif method == 'trapezoidal':
                sim = trapezoidal_series(g_proposed, k, y0, t_window)
            #elif method == 'analytical':
                #t_rel = t_window - t_window[0]
                #sol = np.asarray([Solution(t, g_proposed, k, y0) for t in t_rel])
                #sim = sol.T
            else:
                raise ValueError("unknown method")
            dist = np.linalg.norm(obs_vec - sim)
            
            if dist <= eps_t:  # Accept
                temp_particles.append(g_proposed)
                temp_distances.append(dist)
                acc += 1
        
        # Check if we got any particles
        if acc == 0:
            if verbose:
                print(f"[ABC-{method}] WARNING: No particles accepted!")
            # If no particles in first generation, fail gracefully
            if t == 0:
                # Return default values
                return 0.3, 0.1, 1e-10, np.array([0.3])
            else:
                # For later generations, keep previous particles and weights
                actual_particles = prev_actual_particles
                particles[t, :actual_particles] = particles[t-1, :actual_particles]
                weights[t, :actual_particles] = weights[t-1, :actual_particles]
                # Don't update prev_actual_particles
                continue
        
        # Update actual number of particles
        if t == 0:
            prev_actual_particles = acc
        actual_particles = acc
        
        # Store accepted particles
        for i in range(acc):
            particles[t, i] = temp_particles[i]
            distances[i] = temp_distances[i]
        
        # Weight update (after all particles collected)
        if t == 0:
            # First generation: uniform weights
            weights[t, :acc] = 1.0 / acc
        else:
            # Later generations: importance weights
            temp_weights = np.zeros(acc)
            for j in range(acc):
                # Compute kernel density for this particle
                kernel_vals = np.exp(-0.5 * ((particles[t, j] - prev_particles) / sigma)**2)
                denom = np.sum(prev_weights * kernel_vals)
                temp_weights[j] = 1.0 / max(denom, 1e-12)
            
            # Normalize weights
            weight_sum = np.sum(temp_weights)
            if weight_sum > 0:
                weights[t, :acc] = temp_weights / weight_sum
            else:
                weights[t, :acc] = 1.0 / acc
        
        # Update prev_actual_particles for next iteration
        prev_actual_particles = acc
        
        if verbose and attempts > 0:
            ar = acc/attempts
            print(f"[ABC-{method}] Accepted {acc}/{attempts} (rate: {ar:.3g})")
    
    # Posterior summary using actual particles
    if actual_particles == 0:
        # No particles accepted - return default values
        return 0.3, 0.1, 1e-10, np.array([0.3])
    
    final_particles = particles[num_generations-1, :actual_particles]
    final_weights = weights[num_generations-1, :actual_particles]
    
    # Check if weights are valid
    weight_sum = np.sum(final_weights)
    if weight_sum <= 0:
        # If weights sum to zero, use uniform weights
        final_weights = np.ones(actual_particles) / actual_particles
    else:
        final_weights = final_weights / weight_sum  # Normalize
    
    g_mean = np.average(final_particles, weights=final_weights)
    
    # Handle potential numerical issues in variance calculation
    if len(final_particles) > 1:
        g_std = np.sqrt(np.average((final_particles - g_mean)**2, weights=final_weights))
    else:
        g_std = 0.1  # Default std if only one particle
    
    # Evidence proxy using acceptance rate and final epsilon
    # Use prev_actual_particles if no particles accepted in last generation
    final_particle_count = actual_particles if actual_particles > 0 else prev_actual_particles
    evidence = (final_particle_count / max(total_attempts, 1)) / eps_schedule[-1]
    
    if verbose:
        print(f"[ABC-{method}] Done. γ = {g_mean:.3f} ± {g_std:.3f}, evidence ≈ {evidence:.3e}")
    
    return g_mean, g_std, evidence, final_particles

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
    for i in range(n_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size + 1
        
        if end_idx > len(t):
            break
        
        # Extract window data
        t_window = t[start_idx:end_idx]
        y_window = observed_y[start_idx:end_idx]
        yp_window = observed_yp[start_idx:end_idx]
        
        # Make t_window relative to start of window
        t_window_rel = t_window - t_window[0]
        
        window_center = np.mean(t_window)
        window_centers.append(window_center)
        
        print(f"Window {i+1}: t = {t_window[0]:.1f}s to {t_window[-1]:.1f}s")
        
        # Estimate parameters for each method
        # Euler method
        gamma_euler, evidence_euler = compute_model_evidence(
            'euler', k, t_window_rel, y_window, yp_window, noise_level
        )
        euler_gammas.append(gamma_euler)
        euler_evidence.append(evidence_euler)
        print(f"  Euler: γ = {gamma_euler:.3f} (error = {abs(gamma_euler - gamma_true):.3f})")
        
        # Trapezoidal method
        gamma_trap, evidence_trap = compute_model_evidence(
            'trapezoidal', k, t_window_rel, y_window, yp_window, noise_level
        )
        trap_gammas.append(gamma_trap)
        trap_evidence.append(evidence_trap)
        print(f"  Trapezoidal: γ = {gamma_trap:.3f} (error = {abs(gamma_trap - gamma_true):.3f})")
        
        # Generate predictions for this window
        y0_window = np.array([y_window[0], yp_window[0]])
        
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
            quantiles=[80, 60, 40]  # Three-stage ladder
        )
        print(f"  Joint ε schedule: {eps_schedule}")
        
        # Run Euler with joint schedule
        res_eu = abc_smc_estimation(
            'euler', k, t_window_rel, y_window, yp_window,
            eps_schedule=eps_schedule,
            verbose=False
        )
        g_eu_abc, Z_eu_abc = res_eu[0], res_eu[2]
        
        # Run Trapezoidal with SAME schedule
        res_tr = abc_smc_estimation(
            'trapezoidal', k, t_window_rel, y_window, yp_window,
            eps_schedule=eps_schedule,
            verbose=False
        )
        g_tr_abc, Z_tr_abc = res_tr[0], res_tr[2]
        
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

    bayes_factors = [trap_evidence[i] / euler_evidence[i] if euler_evidence[i] > 0 else 1 
                     for i in range(len(euler_evidence))]

    # Build full-length evidence & Bayes-factor signals
    evidence_euler_full = np.zeros_like(t)
    evidence_trap_full = np.zeros_like(t)
    bayes_full = np.zeros_like(t)
    
    for i in range(n_windows):
        s = i * window_size
        e = s + window_size + 1
        evidence_euler_full[s:e] = euler_evidence[i]
        evidence_trap_full[s:e] = trap_evidence[i]
        bayes_full[s:e] = (trap_evidence[i] / euler_evidence[i] if euler_evidence[i] > 0 else np.inf)
    
    # Piece-wise-constant ABC evidence over the whole timeline
    abc_evidence_eu_full = np.zeros_like(t, dtype=float)
    abc_evidence_tr_full = np.zeros_like(t, dtype=float)
    
    for i in range(n_windows):
        s = i * window_size
        e = s + window_size + 1
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
    plt.title('Vinalla ABC (with joint ε schedule)', fontsize=14)
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
    plt.title('Vinalla LAplase-MLE', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 3 - MLE evidence & Bayes factor (with L2 norm evidence)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # (a) Evidence (MLE)
    ax1.step(t, evidence_euler_full, where='post', color='blue', label='Euler (Laplace)')
    ax1.step(t, evidence_trap_full, where='post', color='red', label='Trap (Laplace)')
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
    ax2.set_title('(b) Bayes factor — MLE')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.suptitle('Laplace with MLE: Evidence (L2 norm) & Bayes factor')
    plt.tight_layout()
    plt.show()
    
    # Figure 4 — ABC-SMC evidence & Bayes factor
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # (a) Evidence (ABC-SMC)
    ax1.step(t, abc_evidence_eu_full, where='post', color='cyan', ls='--', label='Euler (ABC)')
    ax1.step(t, abc_evidence_tr_full, where='post', color='orange', ls='--', label='Trap (ABC)')
    ax1.set_yscale('log')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Model evidence')
    ax1.set_title('(a) Evidence — ABC-SMC (joint ε)')
    ax1.legend()
    ax1.grid(True, which='both', alpha=0.3)
    
    # (b) Bayes factor (ABC-SMC)
    bf_abc = abc_evidence_tr_full / abc_evidence_eu_full
    ax2.step(t, bf_abc, where='post', color='green', ls='--', label='Trap/Euler (ABC)')
    ax2.axhline(1.0, color='gray', ls='--', alpha=0.6)
    ax2.set_yscale('log')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Bayes factor')
    ax2.set_title('(b) Bayes factor — ABC-SMC')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)
    
    plt.suptitle('  ABC-SMC: Evidence & Bayes factor (joint ε schedule)')
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
