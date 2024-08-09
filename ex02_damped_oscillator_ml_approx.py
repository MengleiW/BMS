import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import scipy.stats as stats
"""
Use Bayesian methods to compare two different discretization schemes for the 
damped oscillator ODE, given by:

    m*x''(t) + k*x(t) + gma*x'(t) = f(t)
    x(0)  = x0
    x'(0) = v0

where x(t) is the position, x'(t) is the velocity, k is the spring constant, 
gma is the damping coefficient, and f(t) is the external force. Written as 
a system of first-order ODEs, we have:

    y'(t) = F(t, y(t)), y(0) = y0

where y(t) = [x(t), x'(t)] and F(t, y) = [y2(t), 1/m*(f(t) - gma*y2(t) - k*y1(t))].

We assume that gma is a uniform continuous random variable with values in the 
interval [0,1]. All other parameters are known. We wish to compare two different
discretization schemes for the ODE, one using the Forward Euler method and the
other using the trapzoidal rule, using the same time step h. 

(i) The Forward Euler is given by:  

    y_n+1 = y_n + h*F(t_n, y_n)

(ii) The trapezoidal rule is given by:

    y_n+1 = y_n + h/2*[F(t_n, y_n) + F(t_n+1, y_n + h*F(t_n, y_n))]

*Measurements*

We assume that we have noisy measurements of the position x(t) 
subject to independent, Gaussian measurement error with standard deviation 
sigma. We compute the reference position by means a fine-grained solution of 
the ODE. The measurements are taken at times t_meas.

Method:

1. Iterate over the time steps:
    a. At each measurement time, and for each model, compute the one-step model evidence:
        i. Estimate the velocity, using the right hand side of the ODE (assume for now that it is known)
        ii. Compute the maximum likelihood estimate of the damping coefficient
        iii. Compute the Fisher information (Hessian of the log-likelihood)
        iv. Compute the model evidence at the current measurement time
    b. Update the prior model evidence using the one-step model evidence

"""
def forward_euler(t, y0, k, gma, m, f):
    """
    Forward Euler discretization of the damped oscillator ODE

    Inputs:

        t: time discretization

        y0: initial conditions

        k: spring constant

        gma: damping coefficient

        m: mass

        f: external force

    Outputs:

        y: solution of the ODE at the time points t.
    """
    nt = len(t)  # number of time steps
    n = len(y0)  # number of states
    
    # Trivial case: return initial condition
    if nt <= 1:
        return np.array(y0, ndmin=2)
    
    y = np.zeros((nt, n))  # initialize the solution
    y[0,:] = y0
    for i in range(nt-1):
        dt = t[i+1] - t[i]
        F_now = damped_oscillator(t[i], y[i,:], k, gma, m, f)
        y[i+1,:] = y[i,:] + dt*np.array(F_now)
    return y

def trapezoidal_rule(t, y0, k, gma, m, f):
    """
    Trapezoidal rule discretization of the damped oscillator ODE

    Inputs:

        t: time discretization

        y0: initial conditions

        k: spring constant

        gma: damping coefficient

        m: mass

        f: external force

    Outputs:

        y: solution of the ODE at the time points t.
    """
    nt = len(t)  # number of time steps
    # Trivial case: return initial condition
    if nt <= 1:
        return np.array(y0, ndmin=2)
    
    n = len(y0)  # number of states
    y = np.zeros((nt, n))  # initialize the solution
    y[0,:] = y0
    for i in range(nt-1):
        dt = t[i+1] - t[i]
        F_now = damped_oscillator(t[i], y[i,:], k, gma, m, f)
        F_next = damped_oscillator(t[i+1], y[i,:] + dt*np.array(F_now), k, gma, m, f)
        y[i+1,:] = y[i,:] + dt/2*(np.array(F_now) + np.array(F_next))

    return y

def euler_one_step_jacobian(y_prev, y_curr, t_interval, n_t, k, gma, m, f):
    """
    Compute the Jacobian of the Forward Euler discretization of the damped oscillator ODE with respect to the gma parameter.

    Inputs:

        y_prev: state at the previous time step

        y_curr: state at the current time step

        t_interval: time interval between the previous and current time steps

        n_t: number of time steps

        k: spring constant

        gma: damping coefficient

        m: mass

        f: external force
    """
    t = np.linspace(t_interval[0], t_interval[1], n_t+1)  # time step
    for i in range(n_t):
        pass
   
   
    dt = t_interval/n_disc  # time step
    F_now = damped_oscillator(t_interval, y_prev, k, gma, m, f)
    F_next = damped_oscillator(t_interval + dt, y_curr, k, gma, m, f)
    return -dt/2*np.array(F_now) + dt/2*np.array(F_next)


def euler_neg_log_likelihood(y_meas, sgm, t, y0, k, gma, m, f):
    """
    Compute the negative log-likelihood of the measurements given the model outputs
    """
    y_model = forward_euler(t, y0, k, gma, m, f)
    return 0.5*np.sum((y_meas - y_model)**2/sgm**2)


def damped_oscillator(t, y, k, gma, m, f):
    """
    The right-hand side of the damped oscillator ODE
    """
    x, p = y[0], y[1]
    
    dxdt = p
    dpdt = 1/m*(f(t) - gma*p - k*x)

    return [dxdt, dpdt]


# Initial conditions
y0 = [1.0, 0.0]

# System parameters
k = 1  # Spring constant
m = 1 # Mass
f = lambda t: 0  # External force
gmas = np.linspace(0, 1.0, 21)  # random damping coefficient
n_gam = len(gmas)  # number of damping coefficients
# Final time 
T = 10


#
# Plot a sample of realizations of the damped oscillator
# 
fig, ax = plt.subplots()
fig.set_size_inches(6, 3)
t = np.linspace(0, 10, 1000)  # resolution of the reference solution

# Solve the ODE for each value of gma
for gma in gmas:
    y = solve_ivp(damped_oscillator, [0,T], y0, args=(k, gma, m, f), dense_output=True).sol 
    ax.plot(t, y(t)[0], linewidth=0.5, color='black')
ax.set_xlabel('Time')
ax.set_ylabel('Position')

# Compute the reference (true) solution
gma_true = 0.5
y_true = solve_ivp(damped_oscillator, [0,T],y0, args=(k, gma_true, m, f), dense_output=True).sol
ax.plot(t, y_true(t)[0], linewidth=1.5, color='black', label='True solution')

#
# Measurements
#
# Time discretization
n_meas = 21  # number of measurements
t_meas = np.linspace(0, 10, n_meas)  # resolution of the measurements
sgm = 0.1  # standard deviation of the measurement noise
y_meas = y_true(t_meas)[0] + np.random.normal(0, sgm, n_meas)  # position measurements
v_meas = y_true(t_meas)[1] + np.random.normal(0, sgm, n_meas)  # velocity measurements
ax.scatter(t_meas, y_meas, color='red', s=25, label='Measurements')

# Plot violin plot around measurements
data = [y_meas[i] + np.random.normal(0, sgm, 1000) for i in range(n_meas)]
vps = ax.violinplot(data, positions=t_meas, widths=0.1, showmeans=False, showextrema=False, 
                    showmedians=False)
for vp in vps['bodies']:
    vp.set_facecolor('#D43F3A')
    vp.set_alpha(1)
plt.legend()
#plt.grid()
plt.tight_layout()
#plt.savefig('/home/hans-werner/Dropbox/work/research/projects/bayesian_model_selection/notes/figs/ex01_samples_plus_measurements.pdf')
plt.show()


# Time step for the discretization schemes
refinement = 1  # number of time steps between successive measurements
n_disc = refinement*(n_meas-1)  # number of time steps chosen relative to measurement frequency (for convenience)
dt = T/n_disc # time step
t_disc = np.linspace(0, T, n_disc+1)  # time discretization

gma_mle_fe = np.zeros(n_meas)  # Maximum likelihood estimates of the damping coefficient using Forward Euler
gma_mle_trap = np.zeros(n_meas)  # Maximum likelihood estimates of the damping coefficient using the trapezoidal rule
for i,t in enumerate(t_disc):
    #
    # Iterate in time
    #

    # Compute the maximum likelihood estimates of the damping coefficient, given the current measurement
    # Initial condition
    xv_now = np.array(y_meas[i], v_meas[i])
    xv_meas = np.array([y_meas[i+1], v_meas[i+1]])
    t_int = np.array([t, t+dt])
    f_cost = []
    for gma in gmas:
        # Evaluate the negative log likelihood of the measurements given the model outputs
        print('xv_meas:',xv_meas)
        print('xv_now:',xv_now)
        print('t_')
        f_cost.append(euler_neg_log_likelihood(xv_meas, sgm, t_int, xv_now, k, gma, m, f))
    f_cost = np.array(f_cost)
    plt.plot(gmas, f_cost)
    plt.show()
    # Use forward Euler to compute the solution
    

    
    # Compute the maximum likelihood estimates of the damping coefficient, given the current measurement
    

# Solve the ODE using the Forward Euler method
y_fe = forward_euler(t_disc, y0, k, gma_true, m, f)

# Solve the ODE using the trapezoidal rule
y_tr = trapezoidal_rule(t_disc, y0, k, gma_true, m, f)

# Plot the solutions
fig, ax = plt.subplots()
fig.set_size_inches(6, 3)
plt.plot(t_disc, y_fe[:,0], '.-', label='Forward Euler')
plt.plot(t_disc, y_tr[:,0], '.-',label='Trapezoidal rule')
plt.plot(t, y_true(t)[0], label='True solution')
plt.scatter(t_meas, y_meas, color='red', s=25, label='Measurements')
plt.grid()
plt.xlabel('Time')
plt.ylabel('Position')  
#plt.title('True Damping Coefficient Value: {}'.format(gma_true))
plt.legend()
fig.tight_layout()
plt.show()
#plt.savefig('/home/hans-werner/Dropbox/work/research/projects/bayesian_model_selection/notes/figs/ex01_true_solution_and_approximation.pdf')
# Compute the model evidence based on the measurements
#
# 
#
# Evidence recomputed at every measurement
Z_cum_trap = np.zeros(n_meas) # Trapezoidal rule
Z_cum_fe = np.zeros(n_meas)  # Forward Euler

# Evidence computed using by updating formula
Z_up_trap = np.zeros(n_meas) # Trapezoidal rule
Z_up_fe = np.zeros(n_meas) # Forward Euler

# Likelihoods of the measurements given the discretization schemes
log_l_fe = np.zeros((n_meas,n_gam))  # Forward Euler
log_l_trap = np.zeros((n_meas,n_gam))  # Trapezoidal rule
for i in range(n_meas):
    #
    # Iterate over the measurements
    #
    Z_fe_sum = 0
    Z_tr_sum = 0
    for j,gma in enumerate(gmas):
        #
        # Iterate over the damping coefficients
        # 
        # Use forward Euler to compute the solution
        y_fe = forward_euler(t_disc[:i*refinement+1], y0, k, gma, m, f)
        y_tr = trapezoidal_rule(t_disc[:i*refinement+1], y0, k, gma, m, f)

        # Extract the times corresponding to the measurements
        idx = np.arange(0, i*refinement+1, refinement)
        
        # Evaluate model outputs at the measurement times
        y_model_fe = y_fe[idx,0]
        y_model_tr = y_tr[idx,0]

        # Update the sum of the evidence at the current value of gma
        Z_fe_sum += stats.multivariate_normal.pdf(y_model_fe, y_meas[:i+1], sgm*np.eye(i+1))
        Z_tr_sum += stats.multivariate_normal.pdf(y_model_tr, y_meas[:i+1], sgm*np.eye(i+1))
    
        # Compute the cumulative likelihood of the measurements given the discretization schemes for each gma
        
        if i == 0:
            print('i:',i)
            # Initialize the likelihoods
            print(y_model_fe[-1], y_meas[i])
            print(stats.norm.pdf(y_model_fe[-1], y_meas[i], sgm))
            print('verifying log-likelihood calculation')
            print(np.abs(stats.norm.pdf(y_model_tr[-1], y_meas[i], sgm)-np.exp(stats.norm.logpdf(y_model_tr[-1], y_meas[i], sgm))))

            log_l_fe[i,j] = stats.norm.logpdf(y_model_fe[-1], y_meas[i], sgm)
            log_l_trap[i,j] = stats.norm.logpdf(y_model_tr[-1], y_meas[i], sgm)
        else:
            # Update the likelihoods
            #print('j:',j)
            #print(y_model_fe[-1], y_meas[i])
            log_l_fe[i,j] = log_l_fe[i-1,j] + stats.norm.logpdf(y_model_fe[-1], y_meas[i], sgm)
            log_l_trap[i,j] = log_l_trap[i-1,j] + stats.norm.logpdf(y_model_tr[-1], y_meas[i], sgm)
            
    # Compute the cumulative evidence
    Z_cum_fe[i] = Z_fe_sum/len(gmas)
    Z_cum_trap[i] = Z_tr_sum/len(gmas)

    # Compute the evidence using the updating formula
    Z_up_fe[i] = np.sum(np.exp(log_l_fe[i,:]))/n_gam
    Z_up_trap[i] = np.sum(np.exp(log_l_trap[i,:]))/n_gam

# Plot the model evidence as a function of measurement time
fig, ax = plt.subplots(2,1)
fig.set_size_inches(6, 6)
ax[0].plot(t_meas, Z_cum_fe, '.-', label='Forward Euler')
ax[0].plot(t_meas, Z_cum_trap, '.-', label='Trapezoidal rule')
#ax[0].plot(t_meas, Z_up_fe, '--', label='Forward Euler (updating)')
#ax[0].plot(t_meas, Z_up_trap, '--', label='Trapezoidal rule (updating)')
ax[0].grid()
ax[0].set_xlabel('Measurement time')
ax[0].set_ylabel('Model evidence')
ax[0].legend()

ax[1].plot(t_meas,Z_cum_trap/Z_cum_fe, '.-', label='Trapezoidal rule/Forward Euler')
ax[1].grid()
ax[1].set_xlabel('Measurement time')
ax[1].set_ylabel('Bayes factor')
ax[1].legend()
plt.tight_layout()
plt.show()
#fig.savefig('/home/hans-werner/Dropbox/work/research/projects/bayesian_model_selection/notes/figs/ex01_model_evidence.pdf')




# Compute the likelihood of the measurements given the discretization schemes
