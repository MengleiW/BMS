# -*- coding: utf-8 -*-
"""
Description: Determine the model evidence for two classes of atomic models,
    using the MCMC samples of the estimated model parameters (tht) and
    associated samples from the model outputs y = f(tht,M).
   
    Each model output is a 9-dimensional vector representing the difference
    between the computed and the reference NIST energies.
   
    The parameters are 2-dimensional vectors representing a subset of orbital
    scaling parameters with an assumed uniform prior on [0.7,1.3]^2.


    We compute the model evidence using
   
   
    1. Importance sampling
    
    2. Hermonic Mean
    
    3. Reverse important sampling with t-distributions and multivriate normal distribution as auxiliary normalized function.

    4. Two stage umbrella
   
Created on Mon Oct  2 9:34:22 2023

@author: whisk
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import pickle
import scipy.special

import random

def uniform_prior(theta, theta_bounds):
    """
    Description: To find the uniform prior.

    Inputs:
       
        theta: theta_i stands for variable of interest
       
        theta_bounds: (d,2) array whose ith row gives the minimum and maximum
            value of theta_i
       
    Outputs:
   
         prior: (1,1) approximatioin of nomalized prior

       
       
     Modified:
   
         10/07/2023 (Menglei Wang)
           
    """
   
    # Dimensions of the sampled parameter
    N, d = theta.shape
   
    # Area of the region
    A = np.prod(theta_bounds[:,1]-theta_bounds[:,0])
   
    # Determine what samples lie within region
    in_region = np.ones(N)
    for di in range(d):
        in_region *= (theta_bounds[di,0]<=theta[:,0])\
                    *(theta[:,1]<=theta_bounds[di,1])
                   
    # Multiply by Area
    prior = in_region/A
   
    return prior



def gauss_log_likelihood(y, sigma):
    """
    Description: Evaluate the multivariate Gaussian log likelihood function
        associated with model outputs

    Inputs:
       
        y : double, (N,m) array of sample model outputs.
       
        sigma: double >0, the standard diviation of Multivariate normal
            distribution representing the experimental error.
       
       
    Outputs:
   
        l: double, (N,) vector of log-likelihood values
           
           
    Modified:
       
        10/08/2023 (Menglei Wang)
           
    """
    # Data size
    N, m = y.shape
   
    # Gaussian log-likelihood
    ll = -0.5*m*np.log(2*np.pi) - 0.5*m*np.log(sigma**2) \
        - 0.5/(sigma**2)*np.sum(y**2,axis=1)
    
    return ll
   
def HM_FindZ(log_likelihood,y):
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
    # Data size
    N, m = y.shape   
    
    #unlog the log_likehood
    likelihood = np.exp(log_likelihood)
    
    #calculate Zhat
    Zhat = 1/((np.sum(1/likelihood))/N)
    
    return Zhat
   
def Is_FindZ(log_likelihood, prior):
    """
    Description: To find the Z = nomalizing constant using Importat sampling method given liklihood and prior

    Inputs:
       
       
        log_Likelihood: Single, (1,1) , l(y|theta_i, M) where theta_i stands for variable of interest , y stands for the data and
        M is our current model.
       
        prior : Single, (1,N) , g(theta_i) where theta_i stands for variable of interest.
       
       
        Outputs:
       
            Zhat: (1,1) approximatioin of nomalized posterior.

           
           
        Modified:
       
            10/05/2023 (Menglei Wang)
           
        """

    # Data size
    N, m = y.shape

    #unlog the log_likehood
    likelihood = np.exp(log_likelihood)
    
    # Summing the product of prior and likelihoo
    phon = np.sum(prior * likelihood) 
    print('phon =', phon)
    # Calculating the normalized posterior
    Zhat = np.sum((prior * likelihood)/phon)
    print('Zhat =', Zhat)
    
    return Zhat
 
def RIS_FindZ(log_likelihood, prior,y):
    """
    Description: To find the Z = nomalizing constant using Importat sampling method given liklihood and prior

    Inputs:
       
       
        log_Likelihood: Single, (1,1) , l(y|theta_i, M) where theta_i stands for variable of interest , y stands for the data and
        M is our current model.
       
        prior : Single, (1,N) , g(theta_i) where theta_i stands for variable of interest.
       
       
        Outputs:
       
            Zhat: (1,1) approximatioin of nomalized posterior.

           
           
        Modified:
       
            10/15/2023 (Menglei Wang)
           
        """

    # Data size
    N, m = y.shape

    #unlog the log_likehood
    likelihood = np.exp(log_likelihood)
    
    # Using multivariate gaussian as my auxiliry function
    #f = -0.5*m*np.log(2*np.pi) - 0.5*m*np.log(sigma**2) \
        #- 0.5/(sigma**2)*np.sum(y**2,axis=1)
        
    #Using t-distributions as auxiliary normalized function
    t = 100
    f = np.log(scipy.special.gammaln((t+1)/2))+(-(t+1)/2)*np.log(1+(m**2)/t)-np.log(((t*math.pi)**0.5)*scipy.special.gammaln(t/2))
    print('f =', f)
    if f < 0:
        f =-f 
    # Calculating the normalized posterior
    Zhat = 1/(np.sum(f/(prior * likelihood))/N)
    print('Zhat =', Zhat)
    
    return Zhat
 
def HMModel_compaire (data,sigma):
    """
    Description: To Compaire models using normalizing constant approximated by Hermonic Mean smapling

    Inputs:
        data: (2,2) The data generated by a MCMC = {'model01': {'tht': tht_for_model01, 'y': y_vals_for_model01 }, 
              'model02': {'tht': tht_for_model02, 'y': y_vals_for_model02 }} 
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.

       
       
        Outputs:
       
            C: (1,1) Ratios between the normalizing constants of two models.

            Graph: A pie graph of the ratio C.
           
        Modified:
       
            10/12/2023 (Menglei Wang)
           
        """
    #extrac data
    tht1 = data['model01']['tht']
    tht2 = data['model02']['tht']
    y1 = data['model01']['y']
    y2 = data['model02']['y']
    theta_bounds = np.array([[0.7,1.3],[0.7,1.3]])
   
    # Data size
    N, m = y1.shape
    
    #Finding Z1
    likelihood1 = gauss_log_likelihood(y1,sigma)
    Z1 = HM_FindZ(likelihood1,y1)
    
   #Finding Z2
    likelihood2 = gauss_log_likelihood(y2,sigma)
    Z2 = HM_FindZ(likelihood2,y2)
    
    #Comparie the nomalizing constant
    C = Z2/Z1
    print('c = ' , C)
    
    #graphing
    plt.pie([Z2,Z1], labels= [Z2,Z1], colors = ["red","blue"])
    plt.annotate('Red is Z2', xy=(-1.1,0.8))
    plt.annotate('Blue is Z1', xy=(-1.1, 0.9)) 
    plt.title('Z1,Z2')
    plt.tight_layout()
    plt.show()
    return C

def ISModel_compaire (data,sigma):
    """
    Description: To Compaire models using normalizing constant approximated by Reverse important smapling

    Inputs:
        data: (2,2) The data generated by a MCMC = {'model01': {'tht': tht_for_model01, 'y': y_vals_for_model01 }, 
              'model02': {'tht': tht_for_model02, 'y': y_vals_for_model02 }} 
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.

       
       
        Outputs:
       
            C: (1,1) Ratios between the normalizing constants of two models.

           
           
        Modified:
       
            10/12/2023 (Menglei Wang)
           
        """
    
  
    
    #extrac data
    tht1 = data['model01']['tht']
    tht2 = data['model02']['tht']
    y1 = data['model01']['y']
    y2 = data['model02']['y']
    
    # Data size
    N, m = y1.shape
    
    #Finding Z1
    prior1 = uniform_prior(tht1)
    likelihood1 = gauss_log_likelihood(y1,sigma,N)
    Z1 = Is_FindZ(N,likelihood1, prior1)
    
   #Finding Z2
    prior2 = uniform_prior(tht2)
    likelihood2 = gauss_log_likelihood(y2,sigma,N)
    Z2 = Is_FindZ(N,likelihood2, prior2)
   
    C = Z2/Z1
    print('c = ' , C)
    return C


def RisModel_compaire (data,sigma):
    """
    Description: To Compaire models using normalizing constant approximated by Hermonic Mean smapling

    Inputs:
        data: (2,2) The data generated by a MCMC = {'model01': {'tht': tht_for_model01, 'y': y_vals_for_model01 }, 
              'model02': {'tht': tht_for_model02, 'y': y_vals_for_model02 }} 
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.

       
       
        Outputs:
       
            C: (1,1) Ratios between the normalizing constants of two models.

            Graph: A pie graph of the ratio C.
           
        Modified:
       
            10/12/2023 (Menglei Wang)
           
        """
    #extrac data
    tht1 = data['model01']['tht']
    tht2 = data['model02']['tht']
    y1 = data['model01']['y']
    y2 = data['model02']['y']
    theta_bounds = np.array([[0.7,1.3],[0.7,1.3]])
   
    # Data size
    N, m = y1.shape
    
    
    #Finding Z1
    log_likelihood1 = gauss_log_likelihood(y1,sigma)
    prior1 = uniform_prior(tht1, theta_bounds)
    Z1 = RIS_FindZ(log_likelihood1,prior1,y1)
    
   #Finding Z2
    log_likelihood2 = gauss_log_likelihood(y2,sigma)
    prior2 = uniform_prior(tht2, theta_bounds)
    Z2 = RIS_FindZ(log_likelihood2,prior2,y2)
    
    #Comparie the nomalizing constant
    C = Z2/Z1
    print('c = ' , C)
    
    #graphing
    plt.pie([Z2,Z1], labels= [Z2,Z1],colors = ["red","blue"])
    plt.annotate('Red is Z2', xy=(-1.1,0.8))
    plt.annotate('Blue is Z1', xy=(-1.1, 0.9)) 
    plt.title('Z1,Z2')
    plt.tight_layout()
    plt.show()
    return C

def Metropolis_hasting(weights,M,m,target_function,proposal_function ):
    """
    Description: Use the Metropolis-Hastings sampler to generate a sample froma Rayleigh distribution.

    Inputs:
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.
       
        m: dimension of the variables.
        
        M: number of samples 
        
        Target_function:   Fucntion of distribution that we wish to sample from. 
        
        proposal_function: Function that we propse for canidates.
        
       
    Outputs:
   
         Theta1: Sample gather from the target distribution.
         Theta2: Sample gather from the target distribution.
       
       
     Modified:
   
         10/22/2023 (Menglei Wang)
              
    """        

    #Set empty parameters
    theta1 = []
    theta2 = []
    X_t = np.ones(2*m)
    
    #if proposal_function == 'Gaussian':
        #proposal_function = x: np.random.multivariate_normal(x, cov=np.eye(len(x)) * 0.1)

    for i in range(M):
        # Propose a new state from multivariate distribution 
        Y = proposal_function(X_t)
        Y1, Y2 = Y[:m], Y[m:]
        #calculate acceptance rate alpha ratio, reduction due to symmetric proposal distributions.
        r = target_function(Y)/target_function(X_t) * weights[i % len(weights)]
       # print('r=',r)
        
        alpha = np.minimum(1, r)
        #print('alpha=',alpha)
        
        if np.random.random() < alpha:
            X_t = Y
            theta1.append(Y1)
            theta2.append(Y2)
        else:
            
            theta1.append(X_t[:m])
            theta2.append(X_t[m:])
    theta1 = np.array([arr.flatten() for arr in theta1])
    theta2 = np.array([arr.flatten() for arr in theta2])
    return theta1, theta2
    
    
def Umbrella_sampling (NS,data, sigma ,target_function, proposal_function ):   
    """
    Description: To Compaire models using normalizing constant ratio approximated by Multi-stage(M stage) umbrella smapling. 
    Drawing sample using MCMC Metropolis-Hastings sampler

    Inputs:
        
        Ns:   (1,1) number of stages
        
        data: (2,2) The data generated by a MCMC = {'model01': {'tht': tht_for_model01, 'y': y_vals_for_model01 }, 
              'model02': {'tht': tht_for_model02, 'y': y_vals_for_model02 }} 
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.

       Target_function:   Fucntion of distribution that we wish to sample from. 
       
       proposal_function: Function that we propse for canidates.
       
        Outputs:
       
            C: (1,1) Ratios between the normalizing constants of two models.

           
        Modified:
       
            11/8/2023 (Menglei Wang)
           
        """
    #extrac data
    y1 = data['model01']['y']
    y2 = data['model02']['y']
    tht2 = data['model02']['tht']

    # Data size
    N, m = tht2.shape
    weight = np.ones(N)
    
    
    
    #Sampling using mertropolis hasting MCMC method.
    tht1,tht2  = Metropolis_hasting(weight,N,m,target_function,proposal_function)
    theta_bounds = np.array([[0.7,1.3],[0.7,1.3]])
    
    
    
    #Finding Q1
    prior1 = uniform_prior(tht1, theta_bounds)
    likelihood1 = gauss_log_likelihood(y1,sigma)
    q1 = np.array(prior1*likelihood1)
    
    #Finding Q2
    prior2 = uniform_prior(tht2, theta_bounds)
    likelihood2 = gauss_log_likelihood(y2,sigma)
    q2 = np.array(prior2*likelihood2)
    
    
    
    
    #Let q12 and q21 = q3, there for the sum of each denominator become N and cancles.
    epsilon = 1e-10
    q1 =np.log(np.maximum(q1, epsilon))
    q2 = np.log(np.maximum(q2, epsilon))
    
    
    rh = np.exp(scipy.special.logsumexp(q1 - q2))
    
        
    #Update q3
    
    weight = np.abs(q1 - rh * q2)
    
    for i in range(NS):
        #Sampling using mertropolis hasting MCMC method.
        tht1,tht2  = Metropolis_hasting(weight,N,m,target_function,proposal_function)

        #tht2 = Metropolis_hasting(N,m,target_function,proposal_function)
        theta_bounds = np.array([[0.7,1.3],[0.7,1.3]])
        
        
        
        #Finding Z1
        prior1 = uniform_prior(tht1, theta_bounds)
        likelihood1 = gauss_log_likelihood(y1,sigma)
        q1 = np.array(prior1*likelihood1)
        
        #Finding Z2
        prior2 = uniform_prior(tht2, theta_bounds)
        likelihood2 = gauss_log_likelihood(y2,sigma)
        q2 = np.array(prior2*likelihood2)
        
        #Let q12 and q21 = q3, there for the sum of each denominator become N and cancles.
        epsilon = 1e-10
        q1 = np.maximum(q1, epsilon)
        q2 = np.maximum(q2, epsilon)
        
        q1 = np.log(q1)
        q2 = np.log(q2)
        rh = np.exp(scipy.special.logsumexp(q1 - q2))
            
        #Update weight
        
        weight = np.abs(q1 - rh * q2)
    
    
    #nomalize q1 and q2
    Z1 = np.sum(np.exp(q1))
    Z2 = np.sum(np.exp(q2))
    print('Z1=',Z1)
    #h= rh +1
    #Z2=rh/h
    #Z1 = 1/h
    
    
    #plt.plot([q2,q1])
    #plt.title('Z1,Z2')
    #plt.tight_layout()
    #plt.show()  
    #graphing
    plt.pie([Z2,Z1], labels= [Z2,Z1],colors = ["red","blue"])
    
    plt.annotate('Red is Z2', xy=(-1.1,0.8))
    plt.annotate('Blue is Z1', xy=(-1.1, 0.9)) 
    plt.title('Z1,Z2')
    plt.tight_layout()
    plt.show()    
    return 




if __name__ == '__main__':
    atomic_data_pickle_path = 'C:\\Users\\whisk\\atomic_data.pickle'
    #atomic_data_path =  '../data/atomic_data.pickle'
    with open('atomic_data.pickle', 'rb')  as f:
       data = pickle.load(f)
    NS = 10   
    sigma = 0.1
    #HMModel_compaire (data,sigma)
    #RisModel_compaire (data,sigma)
   
    
    #y = data['model01']['y']
    
    tht2 = data['model02']['tht']

    

    
    target_function = lambda x : scipy.stats.multivariate_normal.logpdf(x, cov=np.eye(len(x)) * 0.1)
    
    proposal_function  = lambda x: np.random.multivariate_normal(x, cov=np.eye(len(x)) * 0.1)
    
    #print(y.shape)
    US = Umbrella_sampling (NS,data, sigma ,target_function, proposal_function   )
   
    
    #ll = gauss_log_likelihood(y, sigma)
    #theta = data['model02']['tht']
    
    
    #pp = uniform_prior(theta,theta_bounds)
    #Is_FindZ(ll, pp)
    #print('pp=',data.shape)
   
   
    #ISModel_compaire(data,0.1,9)
