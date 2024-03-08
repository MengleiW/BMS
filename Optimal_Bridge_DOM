import numpy as np
import scipy
import matplotlib.pyplot as plt
#import math
#import pymc3 as pm
#import theano.tensor as tt


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
    f_t = 0  
    #f_t = np.sin(t) 
    return [y[1], f_t - gamma*y[1] - k*y[0]]

def simulate_observed_data(gamma, k, initial_conditions, ts, N, noise_level_s, noise_level_j):
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
    t = np.linspace(ts[0], ts[1], N)
    sol = scipy.integrate.solve_ivp(damped_oscillator, ts, initial_conditions, args=(gamma, k), t_eval=t)
    half = N // 2
    y = sol.y[0]
    yp = sol.y[1]  
    # Apply smooth noise to the first half and jumpy noise to the second half
    noise1 = np.concatenate([np.random.normal(0, noise_level_s,  half),
                             np.random.normal(0, noise_level_j, N -  half)])
    noise2 = np.concatenate([np.random.normal(0, noise_level_s,  half),
                             np.random.normal(0, noise_level_j, N -  half)])
    y += noise1
    yp += noise2
    
    return y, yp

def euler_forward(gammas, k, y, t, h):
    """
    Numerically approximates the solution of the damped oscillator using the Euler forward method for a single time point.

    Inputs:
      gammas: Array, range of damping coefficients.
      k: Float, stiffness coefficient.
      y: 2D Array, initial states of the system for each gamma (position and velocity).
      t: Float, single time point.
      h: Float, time step.

    Outputs:
      y: 2D Array, where each row corresponds to the system states (position and velocity) for a specific gamma.
    """

    for i, gamma in enumerate(gammas):
        print(y[i,:])
        y[i,:] += h * np.array(damped_oscillator(t, y[i, :], gamma, k)) 
        

    return y

def trapezoidal_method(gammas, k, y, t, h):
    """
    Numerically approximates the solution of the damped oscillator using the trapezoidal method for a single time point.

    Inputs:
      gammas: Array, range of damping coefficients.
      k: Float, stiffness coefficient.
      y: 2D Array, initial states of the system for each gamma (position and velocity).
      t: Float, single time point.
      h: Float, time step.

    Outputs:
      y: 2D Array, where each row corresponds to the system states (position and velocity) for a specific gamma.
    """
    for i, gamma in enumerate(gammas):
        f_n = np.array(damped_oscillator(t, y[i, :], gamma, k))
        y_pred = y[i, :] + h * f_n
        f_n_plus_1 = np.array(damped_oscillator(t + h, y_pred, gamma, k))
        y[i, :] = y[i, :] + h/2 * (f_n + f_n_plus_1)  
         

    return y


def calculate_combined_log_likelihood(simulated, observed_y, observed_yp, sigma_y, sigma_yp):
    """
    Computes the combined log-likelihood of the observed position and velocity data for a given simulated dataset.
    And since we are intrest both
    Inputs:
        simulated: Array, the simulated data from the model.
        observed_y: Array, observed data for position (y).
        observed_yp: Array, observed data for velocity (y').
        sigma_y: Float, standard deviation of error in position measurements.
        sigma_yp: Float, standard deviation of error in velocity measurements.

    Outputs:
        log_likelihood: Float, the combined log-likelihood value of the observed data given the simulated model outputs.

    """
    ll = np.zeros(simulated.shape[0])

    for i in range(simulated.shape[0]):
        residuals_y = observed_y - simulated[i,  0]
        residuals_yp = observed_yp - simulated[i, 1]

        ll_y = np.sum(scipy.stats.norm.logpdf(residuals_y, scale=sigma_y))
        ll_yp = np.sum(scipy.stats.norm.logpdf(residuals_yp, scale=sigma_yp))

        ll[i] = ll_y + ll_yp

    return ll

def HM_FindZ(log_likelihood, N):
    """
    Description: To find the Z = nomalizing constant using Importat sampling method given liklihood and prior

    Inputs:
       
       
        log_Likelihood: Single, (1,1) , l(y|theta_i, M) where theta_i stands for variable of interest , y stands for the data and
        M is our current model.
       
       
        Outputs:
       
            Zhat: (1,1) approximatioin of nomalized posterior.

           
           
        Modified:
       
            10/05/2023 (Menglei Wang)
           
        """
   
    log_Zhat = scipy.special.logsumexp(log_likelihood) /N
    
    
    # Exponentiate to get Zhat, if necessary. Otherwise, return log_Zhat based on use case.
    Zhat = 1/np.exp(log_Zhat)

    return Zhat

def Saved_DATA(method, gammas, initial_conditions, T, k):
    """
    Simulates and saves data for a range of parameter values over specified time points using a numerical method.

    Inputs:
        method: Function, the numerical method used for simulating the model (e.g., Euler forward or trapezoidal).
        gammas: Array, range of parameter values (e.g., damping coefficients) to be tested.
        initial_conditions: Array, initial state of the system (usually includes initial position and velocity).
        T: Array, time points for which the data is to be simulated.
        k: Float, a parameter of the system (e.g., stiffness coefficient in a damped oscillator model).
        
    Outputs:
        data: 3D Array, where each element contains the simulated system states for each value of gamma at each time point.
    """
    
    data = np.zeros((len(gammas), len(T), 2))
    data[:, 0, :] = initial_conditions 
    #ll = np.zeros( len(T))
    
    for i, t in enumerate(T[1:], start=1): 
        h = t - T[i - 1]
        data[:, i, :] = method(gammas, k, data[:, i - 1, :], t, h)
        #combined_log_likelihood = calculate_combined_log_likelihood(data[:, i, :], observed_y, observed_yp, sigma_y, sigma_yp)
        #ll[i]=combined_log_likelihood

    return  data#, ll
def Slove_Z(data,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp):
    """
    Computes the marginal likelihood estimates (Z) for specified checkpoints with data saved.
    
    Inputs:
        data: 3D Array, simulated data for each gamma over time.
        check_points: List, indices of time points in 'T' at which to calculate Z.
        gammas: Array, range of damping coefficients tested.
        observed_y: Array, observed data for position (y).
        observed_yp: Array, observed data for velocity (y').
        sigma_y: Float, standard deviation of error in position measurements.
        sigma_yp: Float, standard deviation of error in velocity measurements.
    
    Outputs:
        results: List, contains the marginal likelihood estimates (Z) for each checkpoint.
    """
        
    results = []
    for i in range(N):#check_points:
        combined_log_likelihood = calculate_combined_log_likelihood(data[:, i, :], observed_y, observed_yp, sigma_y, sigma_yp)
        Z = HM_FindZ(combined_log_likelihood, len(gammas))
        results.append(Z)
    
    return results 


def iterative_Z (data,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest):
    """
    Computes the marginal likelihood estimates (Z) using P(y_{t+1},...,y_1|M) = P(y_{t+1}|y_t,...,y_1,M)P(M | y_t,...,y_1)
    
    Inputs:
        data: 3D Array, simulated data for each gamma over time.
        check_points: List, indices of time points in 'T' at which to calculate Z.
        gammas: Array, range of damping coefficients tested.
        observed_y: Array, observed data for position (y).
        observed_yp: Array, observed data for velocity (y').
        sigma_y: Float, standard deviation of error in position measurements.
        sigma_yp: Float, standard deviation of error in velocity measurements.
    
    Outputs:
        results: List, contains the marginal likelihood estimates (Z) for each checkpoint.
    """
    Zhat = Slove_Z(data,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp)[Timepoint_of_interest]
    
    results = []
    
    # Assuming you want to use this index to start further calculations
    for i in range(Timepoint_of_interest, N):
        combined_log_likelihood = calculate_combined_log_likelihood(data[:, i, :], observed_y, observed_yp, sigma_y, sigma_yp)
        #if i % 5 == 0:
            #Zhat = HM_FindZ(combined_log_likelihood, len(gammas))
            #results.append(Zhat)
        #else:
            
        log_Zhat = np.log(Zhat)
        #log_Zhat = log_Zhat + scipy.special.logsumexp(combined_log_likelihood) /N
        Zhat = log_Zhat + np.sum(np.exp(combined_log_likelihood))*(gammas[1] - gammas[0])
        #Zhat = 1/np.exp(log_Zhat)
        results.append(Zhat)

    return results  
        


def Metropolis_hasting(M,m,target_function,proposal_function ):
    """
    Metropolis-Hasting algorithm for sampling from a target distribution.

    Parameters:
    - M: Number of samples to generate.
    - m: The dimension of the parameter space.
    - target_function: Function to compute the log-likelihood of a state.
    - proposal_function: Function to propose a new state given the current state.

    Returns:
    - A list of sampled states from the target distribution.
       
     Modified:
   
         11/09/2023 (Menglei Wang)
    """ 

    #Set empty parameters
    theta = []
    
    X_t = np.zeres(m)
    
    #if proposal_function == 'Gaussian':
        #proposal_function = x: np.random.multivariate_normal(x, cov=np.eye(len(x)) * 0.1)

    for i in range(M):
        # Propose a new state from multivariate distribution 
        Y = proposal_function(X_t)
        
        #calculate acceptance rate alpha ratio, reduction due to symmetric proposal distributions.
        r = target_function(Y)/target_function(X_t) #* weights
       # print('r=',r)
        
        alpha = np.minimum(1, r)
        #print('alpha=',alpha)yh
        
        if np.random.random() < alpha:
            X_t = Y
            
        theta.append(X_t)
        theta= np.array(theta)
    return theta

def target_function(method,gammas,initial_conditions,T, k, observed_y, observed_yp, sigma_y, sigma_yp):
    """
    Simulates and saves data for a range of parameter values over specified time points using a numerical method.

    Inputs:
        method: Function, the numerical method used for simulating the model (e.g., Euler forward or trapezoidal).
        gammas: Array, range of parameter values (e.g., damping coefficients) to be tested.
        initial_conditions: Array, initial state of the system (usually includes initial position and velocity).
        T: Array, time points for which the data is to be simulated.
        k: Float, a parameter of the system (e.g., stiffness coefficient in a damped oscillator model).
        
        observed_y: Array, observed data for position (y).
        observed_yp: Array, observed data for velocity (y').
        sigma_y: Standard deviation of error in position measurements.
        sigma_yp: Standard deviation of error in velocity measurements.
        
        
    Outputs:
        target_function:  where each element contains the simulated system states for each value of gamma at given time point.
    """
    simulated_data = np.zeros((len(T),2))
    simulated_data[0,:] = initial_conditions
    
    for i, t in enumerate(T[1:], start=1): 
        h = t - T[i - 1]
        simulated_data [i,:] = method(gammas, k, simulated_data[i - 1,:], t, h)
        
    ll = calculate_combined_log_likelihood(simulated_data, observed_y, observed_yp, sigma_y, sigma_yp)
    target_function = np.sum(ll)
    return  target_function


def OptimalBridge (method,initial_conditions,T, k, data,N,N1,N2, check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest):
    
    """
    Optimal Bridge Sampling uses Metropolis-Hasting aproximate Z and iteratively updates the aproximation.

    Inputs:
        method: Function, the numerical method used for simulating the model (e.g., Euler forward or trapezoidal).
        gammas: Array, range of parameter values (e.g., damping coefficients) to be tested.
        initial_conditions: Array, initial state of the system (usually includes initial position and velocity).
        T: Array, time points for which the data is to be simulated.
        k: Float, a parameter of the system (e.g., stiffness coefficient in a damped oscillator model).
        h: Float, time step.
        observed_y: Array, observed data for position (y).
        observed_yp: Array, observed data for velocity (y').
        sigma_y: Standard deviation of error in position measurements.
        sigma_yp: Standard deviation of error in velocity measurements.
       
    Outputs:
   
         Rhat: (0,0) The ratio between Z1 and Z2

       
       
     Modified:
   
         10/07/2023 (Menglei Wang)
           
    """
    Z = []
    Zhat = Slove_Z(data,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp)[Timepoint_of_interest]
    
    
    target_function_Post =  target_function(method, gammas,initial_conditions,T, k, observed_y, observed_yp, sigma_y, sigma_yp)/number_of_gammas
    proposal_function_Post  = lambda x: np.random.multivariate_normal(x, cov=np.eye(len(x)) *0.3)
    target_function_phat = lambda x : scipy.stats.multivariate_normal.logpdf(x, cov=np.eye(len(x)) * 0.3)
    proposal_function_phat  = lambda x: np.random.multivariate_normal(x, cov=np.eye(len(x)) * 0.3)
    for i in range (N-1):
         #Taking sampling using Metropolis Hasting algrithm. 
         tht2 = Metropolis_hasting(N,target_function_Post,proposal_function_Post )
         tht1 = Metropolis_hasting(N,target_function_phat,proposal_function_phat )
         
         
         #Finding Q11
         q11 =  target_function(method,tht1,initial_conditions,T, k, observed_y, observed_yp, sigma_y, sigma_yp)#[]
         #print('l1=',likelihood1)
         
         
         #Finding Q12
         q12 = target_function(method,tht2, initial_conditions,T, k, observed_y, observed_yp, sigma_y, sigma_yp)
         #print('q12=',q12)
         
         #Finding Q21
         q21 = scipy.stats.multivariate_normal.logpdf(tht1, cov=np.eye(len(tht1)) * 0.3)
         #print('q21=',q21)
        
         #Finding Q22
        
         q22 = scipy.stats.multivariate_normal.logpdf(tht2, cov=np.eye(len(tht2)) * 0.3)
         #print('l2=',likelihood2)
        
         #epsilon = 1e-10
         #q11 = np.maximum(q11, epsilon)
         #q12 = np.maximum(q12, epsilon)
         #q21 = np.maximum(q21, epsilon)
         #q22 = np.maximum(q22, epsilon)
         
         
         Q1 = np.logaddexp.reduce(np.log(q11)) - np.logaddexp.reduce(np.log(N1*q11 + N2*Zhat*q21))+N1
         Q2 = np.logaddexp.reduce(np.log(q22)) - np.logaddexp.reduce(np.log(N1*q12 + N2*Zhat*q22))+N2
         print('Q1=',Q1)
         print('Q2=',Q2)
         
         
         
         
         
         zhat = np.exp(Q1 - Q2)
         Z.append(zhat*Z)

    return Z

#def intergrate_LL ():
    
    
if __name__ == '__main__':   
    
    #print('debug=',combined_log_likelihood)
    # Parameters 
    m = 1
    k = 0.5
    initial_conditions = [1, 0]
    number_of_gammas = 10
    gammas = np.linspace(0, 1, 10)
    T = np.linspace(0, 10, 100)
    check_points = [10,50,80]
    
    Timepoint_of_interest=0
    N = 100
    N1 = 100
    N2 = 100
    sigma_y = 0.3 
    sigma_yp = 0.1
    noise_level_s = 0.0001
    noise_level_j = 0.01
    
    
    gamma = 0.1
    observed_y, observed_yp = simulate_observed_data(gamma, k, initial_conditions, [0,100], N, noise_level_s, noise_level_j)
    
    # Compute marginal likelihoods for each method
    data_e = Saved_DATA(euler_forward, gammas, initial_conditions, T, k) 
    #print("Z_e = ",Z_e)
    data_t = Saved_DATA(trapezoidal_method, gammas, initial_conditions, T, k)
    #print("Z_t = ",Z_t)
    
    #Z_e = Slove_Z(data_e,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp)
    #Z_t = Slove_Z(data_t,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp)
    #Z_e = iterative_Z(data_e,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest)
    #Z_t = iterative_Z(data_t,check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest)
    Z_e = OptimalBridge (euler_forward, initial_conditions,T, k, data_e,N,N1,N2, check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest)
    Z_t = OptimalBridge (trapezoidal_method,initial_conditions,T, k, data_t ,N,N1,N2, check_points,gammas, observed_y, observed_yp, sigma_y, sigma_yp,Timepoint_of_interest)
    
    print("Z_e contents:", Z_e)
    print("Length of Z_e:", len(Z_e))
    print("Z_t contents:", Z_t)
    print("Length of Z_t:", len(Z_t))
    
    #graphing
    #checkpoints = list(range(1, len(Z_e) + 1)) 
    checkpoints = list(range(N))

    plt.plot(checkpoints, Z_e, label='Z_e', marker='o')  
    plt.plot(checkpoints, Z_t, label='Z_t', marker='x')  
    
    plt.xlabel('Checkpoints')
    plt.ylabel('Values')
    plt.title('Line Plot of Z_e and Z_t over Checkpoints')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.pie([Z_e[-1], Z_t[-1]], labels=["Z_e", "Z_t"], colors=["red", "blue"])  
    plt.annotate('Red is Z_e', xy=(-1.1,0.8))
    plt.annotate('Blue is Z_t', xy=(-1.1, 0.9)) 
    plt.title('Z_t,Z_e')
    plt.tight_layout()
    plt.show() 
