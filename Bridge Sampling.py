"""
This file provides a single function that uses the RIS to find initial Rhat and iterateion follows Optimal Bridge Samling
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.special
import BMS



def Metropolis_hasting(weights,M,m,target_function,proposal_function ):
    """
    Description: Use the Metropolis-Hastings sampler to generate a sample from a Rayleigh distribution.

    Inputs:
       
        sigma: double >0, the standard deviation of Multivariate normal
               distribution representing the experimental error.
       
        m: the dimension of the variables.
        
        M: number of samples 
        
        Target_function:   Fucntion of distribution that we wish to sample from. 
        
        proposal_function: Function that we propose for candidates.
        
       
    Outputs:
   
         Theta1: Sample gathered from the target distribution.
         Theta2: Sample gathers from the target distribution.
       
       
     Modified:
   
         11/09/2023 (Menglei Wang)
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
        r = target_function(Y)/target_function(X_t) #* weights
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


def OptimalBridge (data,sigma, NS):
    
    """
    Description: To find the uniform prior.

    Inputs:
       
        Ns:   (1,1) number of stages
        
        data: (2,2) The data generated by a MCMC = {'model01': {'tht': tht_for_model01, 'y': y_vals_for_model01 }, 
              'model02': {'tht': tht_for_model02, 'y': y_vals_for_model02 }} 
       
        sigma: double >0, the standard diviation of Multivariate normal
               distribution representing the experimental error.
       
    Outputs:
   
         Rhat: (0,0) The ratio between Z1 and Z2

       
       
     Modified:
   
         10/07/2023 (Menglei Wang)
           
    """
    
    #extrac data
    tht1 = data['model01']['tht']
    tht2 = data['model02']['tht']
    y1 = data['model01']['y']
    y2 = data['model02']['y']
    theta_bounds = np.array([[0.7,1.3],[0.7,1.3]])
    N, m = tht2.shape
    N1,D1 = tht1.shape
    N2,D2 = tht1.shape
    target_function = lambda x : scipy.stats.multivariate_normal.logpdf(x, cov=np.eye(len(x)) * 0.1)
    
    proposal_function  = lambda x: np.random.multivariate_normal(x, cov=np.eye(len(x)) * 0.1)
    
    #Finding Q11
    prior1 = BMS.uniform_prior(tht1, theta_bounds)
    likelihood1 = BMS.gauss_log_likelihood(y1,sigma)
    q11 = np.array(prior1*likelihood1)
    
    #Finding Q22
    prior2 = BMS.uniform_prior(tht2, theta_bounds)
    likelihood2 = BMS.gauss_log_likelihood(y2,sigma)
    q22 = np.array(prior2*likelihood2)
    
    #Finding Q12
    q12 = np.array(prior2*likelihood1)
    
    #Finding Q21
    q21 = np.array(prior1*likelihood2)
    
    #Caculating the initial rhat
    rhat =  1/BMS.RisModel_compaire(data, sigma)
    print('rhat=', rhat)
    
    #Update q3
    
    weights = np.logaddexp.reduce(q12*q22) - np.logaddexp.reduce(N1*q12 - N2*rhat*q22)
    
    
    for i in range (NS):
         #Taking sampling using Metropolis Hasting algrithm. 
         tht1, tht2 = Metropolis_hasting(weights,N,m,target_function,proposal_function )
         
         
         #Finding Q11
         prior1 = BMS.uniform_prior(tht1, theta_bounds)
         likelihood1 = BMS.gauss_log_likelihood(y1,sigma)
         q11 = np.array(prior1*likelihood1)
         #print('l1=',likelihood1)
         
         #Finding Q22
         prior2 = BMS.uniform_prior(tht2, theta_bounds)
         likelihood2 = BMS.gauss_log_likelihood(y2,sigma)
         q22 = np.array(prior2*likelihood2)
         #print('l2=',likelihood2)
         
         #Finding Q12
         q12 = np.array(prior2*likelihood1)
         #print('q12=',q12)
         
         #Finding Q21
         q21 = np.array(prior1*likelihood2) 
         #print('q21=',q21)
        
         epsilon = 1e-10
         q11 = np.maximum(q11, epsilon)
         q12 = np.maximum(q12, epsilon)
         q21 = np.maximum(q21, epsilon)
         q22 = np.maximum(q22, epsilon)
         
         
         Q1 = np.logaddexp.reduce(np.log(q12)) - np.logaddexp.reduce(np.log(N1*q12 + N2*rhat*q22))
         Q2 = np.logaddexp.reduce(np.log(q21)) - np.logaddexp.reduce(np.log(N1*q11 + N2*rhat*q21))
         print('Q1=',Q1)
         print('Q2=',Q2)
         
         
         
         
         
         rhat = np.exp(Q1 - Q2)
         #Update q3
         
         weights = np.logaddexp.reduce(q12*q22) - np.logaddexp.reduce(N1*q12 - N2*rhat*q22)
         #print('weights=',weights)
        
    #graphing
    Q1 = np.sum(np.exp(q11))
    Q2 = np.sum(np.exp(q22))
    print('Q1=',Q1)
    print('Q2=',Q2)
    print('rhat=', rhat)
    
    plt.pie([Q2,Q1], labels= [Q2,Q1], colors = ["red","blue"],autopct='%1.1f%%')
    plt.annotate('Red is Z2', xy=(-1.1,0.8))
    plt.annotate('Blue is Z1', xy=(-1.1, 0.9)) 
    plt.title('Rhat')
    plt.tight_layout()
    plt.show()

    return rhat


if __name__ == '__main__':
    atomic_data_path = 'C:\\Users\\whisk\\atomic_data.pickle'
    #atomic_data_path =  '../data/atomic_data.pickle'
    with open('atomic_data.pickle', 'rb')  as f:
       data = pickle.load(f)
    NS = 10  
    sigma = 0.1
    #HMModel_compaire (data,sigma)
    #RisModel_compaire (data,sigma)
   
    
    y = data['model01']['y']
    
    tht2 = data['model02']['tht']
    
    OptimalBridge (data,sigma, NS)
    
