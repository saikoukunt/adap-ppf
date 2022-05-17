import numpy as np
from numpy import linalg as la
import math
from scipy import linalg
from sympy import *

def ppf(N, delta, expr, param_sym, target_sym, init_params, F, Q, data, spikes):
    """
    Applies a point-process filter to the input data.
    
    Parameters
    ----------
    N : int
        length (duration) of the input data
    delta : int
        timestep of the input data
    expr : SymPy expression
        firing rate equation for given neurons
    param_sym: list
        list of SymPy symbols for desired parameters
    target_sym: list 
        list of SymPy symbols for desired target variables
    init_params: np.array
        initial conditions for parameters. Must have shape (# of neurons, # of params per neuron)
    F: np.array
        system evolution matrix for parameter evolution. Must have shape (# of params per neuron, # of params per neuron)
    Q: np.array
        white noise covariance matrix for parameter evolution. Must have shape (# of params per neuron, # of params per neuron)
    data: np.array
        values of target variables. Must have shape(N, # of target variables)
    spikes: np.array
        spiking data. Must have shape(# of neurons, # of target variables)
        
    Returns
    -------
    params: np.array
        time evolution of parameter mean
    W: np.array
        time evolution parameter variance
        
    """
    
    num_neurons = spikes.shape[0]
    assert init_params.shape[0] == num_neurons, "Number of neurons must be first dimension of init_params and spikes!"
    num_param = len(expr.free_symbols) - len(target_sym)
    assert init_params.shape[1] == num_param, "init_params must have dimensions: # of neurons X # of parameters per neuron!"
    assert F.shape[0] == num_param and F.shape[1] == num_param, "F must be a square matrix with dimension: # of parameters"
    assert Q.shape[0] == num_param and Q.shape[1] == num_param, "Q must be a square matrix with dimension: # of parameters"
    assert len(target_sym) == data.shape[1], "data must contain values for all target variables!"
    
    params = np.zeros((num_neurons, N, num_param))
    params[:,0,:] = init_params
    
    W = np.zeros((num_neurons, N, num_param, num_param))
    W[:,0,:,:] = Q
    
    param_sym = list(param_sym)
    target_sym = list(target_sym)
    sym = param_sym + target_sym
    
    log_expr = log(expr)
    log_expr = simplify(log_expr)
    
    jac1 = Matrix([log_expr]).jacobian(param_sym)
    jac2 = jac1.jacobian(param_sym)

    calc_pd1 = lambdify(sym, jac1)
    calc_pd2 = lambdify(sym, jac2)
    calc_rate = lambdify(sym, expr)
    
    for t in range(1,N):
        for c in range(num_neurons):
            # calculate intermediate parameter mean and variance
            thet = F@(params[c,t-1].reshape((1,num_param)).T)
            var = F@W[c,t-1]@np.transpose(F)+Q

            # calculate partial derivatives
            args = list(thet.flatten())
            for i in range(len(target_sym)):
                args.append(data[t,i])
                
            pd1 = calc_pd1(*args)
            pd2 = calc_pd2(*args)
            
            rate = calc_rate(*args)
            W[c,t,:,:] = la.inv(la.inv(var) + rate*delta*pd1.T@pd1 - (spikes[c,t]-rate*delta)*pd2)
            params[c,t,:] = (thet + W[c,t,:,:]@(pd1.T*(spikes[c,t]-rate*delta)))[:,0]
            
    return params, W


def ofc_ppf(N, delta, expr, param_sym, target_sym, init_params, Q, Q_kin, A, B, L, x_targ, spikes):
    """
    Applies a point-process filter to the input data.
    
    Parameters
    ----------
    N : int
        length (duration) of the input data
    delta : int
        timestep of the input data
    expr : SymPy expression
        firing rate equation for given neurons
    param_sym: list
        list of SymPy symbols for desired parameters
    target_sym: list 
        list of SymPy symbols for desired target variables
    init_params: np.array
        initial conditions for parameters. Must have shape (# of neurons, # of params per neuron)
    Q: np.array
        white noise covariance matrix for parameter evolution. Must have shape (# of params per neuron, # of params per neuron)
    Q_kin: np.array
        white noise covariance matrix for kinematic evolution, computed using fit_ofc_model. Must have shape (# of target variables, # of target variables)
    A, B, L: np.arrays
        optimal feedback control parameters, computed using fit_ofc_model
    x_targ: np.array
        target kinematics. Must have shape (1, # of target variables)
    spikes: np.array
        spiking data. Must have shape(# of neurons, # of target variables)
        
    Returns
    -------
    x: np.array
        decoded kinematics
    params: np.array
        time evolution of parameter mean
    W: np.array
        time evolution parameter variance
    """
    # verify input dimensions
    num_neurons = spikes.shape[0]
    num_param = len(param_sym)
    num_target = len(target_sym)
    
    assert init_params.shape[0] == num_neurons, "Number of neurons must be first dimension of init_params and spikes!"
    assert init_params.shape[1] == num_param, "init_params must have dimensions: # of neurons X # of parameters per neuron!"
    assert Q.shape[0] == num_param and Q.shape[1] == num_param, "Q must be a square matrix with dimension: # of parameters"
    assert Q_kin.shape[0] == num_target and Q_kin.shape[1] == num_target, "Q_kin must be a square matrix with dimension: # of target variables"

    # initialize kinetic PPF 
    x =  np.zeros((N, num_target))
    x_var = np.zeros((N, num_target, num_target))
    x_int = x =  np.zeros((N, num_target))
    x_var[0] = Q_kin
    
    # initialize parameter PPF
    params = np.zeros((num_neurons, N, num_param))
    params[:,0,:] = init_params
    W = np.zeros((num_neurons, N, num_param, num_param))
    W[:,0,:,:] = Q
    
    # create sympy lambda functions for rate and partial derivatives of log 
    param_sym = list(param_sym)
    target_sym = list(target_sym)
    sym = param_sym + target_sym
    
    log_expr = log(expr)
    log_expr = simplify(log_expr)
    
    jac1 = Matrix([log_expr]).jacobian(param_sym)
    jac2 = jac1.jacobian(param_sym)
    jac1_kin = Matrix([log_expr]).jacobian(target_sym)
    jac2_kin = jac1_kin.jacobian(target_sym)
            
    calc_pd1 = lambdify(sym, jac1)
    calc_pd2 = lambdify(sym, jac2)
    calc_pd1_kin = lambdify(sym, jac1_kin)
    calc_pd2_kin = lambdify(sym, jac2_kin)
    calc_rate = lambdify(sym, expr)
    
    # decoding
    for t in range(1,N):
        # estimate kinematics 
        # prediction step
        pred = A@x[t-1]
        pred_var = A@x_var[t-1]@A.T + Q_kin
        
        # update step
        # variance 
        pred_var = la.inv(pred_var)
        for c in range(num_neurons):
            thet = params[c,t-1,:].reshape((1,num_param)).T
            args = list(thet.flatten())
            for i in range(len(target_sym)):
                args.append(pred[i].flatten())
            
            rate = calc_rate(*args)
            pd1_kin = calc_pd1_kin(*args)
            pd2_kin = calc_pd2_kin(*args)

            pred_var += rate*delta*pd1_kin.T@pd1_kin - (spikes[c,t]-rate*delta)*pd2_kin
            pred_var[np.abs(pred_var) < 1*10**-10] = 0  # size check to ignore very small numbers

        pred_var = la.inv(pred_var)
        x_var[t] = pred_var

        
        # mean
        for c in range(num_neurons):
            thet = params[c,t-1,:].reshape((1,num_param)).T
            args = list(thet.flatten())
            for i in range(len(target_sym)):
                args.append(pred[i].flatten())
            
            rate = calc_rate(*args)
            pd1_kin = calc_pd1_kin(*args)
            
            pred += (pred_var@pd1_kin.T*(spikes[c,t]-rate*delta))[:,0]
        x[t] = pred
        
        # calculate intended kinematics
        x_int[t,:] = ((A - B@L)@(x[t].reshape((1,num_target)).T) + B@L@x_targ.T)[:,0]
        
        # decoder params for each neuron
        for c in range(num_neurons):
            # calculate intermediate parameter mean and variance
            thet = params[c,t-1].reshape((1,num_param)).T
            var = W[c,t-1]+Q

            # calculate partial derivatives
            args = list(thet.flatten())
            for i in range(len(target_sym)):
                args.append(x_int[t,i])
                
            pd1 = calc_pd1(*args)
            pd2 = calc_pd2(*args)
            
            rate = calc_rate(*args)
            W[c,t,:,:] = la.inv(la.inv(var) + rate*delta*pd1.T@pd1 - (spikes[c,t]-rate*delta)*pd2)
            params[c,t,:] = (thet + W[c,t,:,:]@(pd1.T*(spikes[c,t]-rate*delta)))[:,0]
            
    return x, params, W

def fit_vel_model(data_x, data_y):
    """
    Maximum likelihood estimation of the system evolution matrix and white noise covariance for a kinematic evolution model.
    
    Parameters
    ----------
    data_x: np.array
        x-direction velocity
    data_y: np.array
        y-direction velocity, must have same length as data_x
        
    Returns
    -------
    Av: np.array
        system evolution matrix
    Wv: np.array
        white noise covariance
    """
    
    # velocity state transition model
    Av = np.zeros((2,2))  # kinematics evolution matrix
    Wv = np.zeros((2,2))  # white noise
    
    # calculate and tile velocity kinematics
    vx1 = np.diff(data_x[:-1], 1); vx2 = np.diff(data_x[1:], 1)
    vy1 = np.diff(data_y[:-1], 1); vy2 = np.diff(data_y[1:], 1)
    v1 = np.vstack((vx1,vy1))
    v2 = np.vstack((vx2,vy2))
    
    # maximum likelihood estimation 
    Av = v2@(v1.T)@(la.inv(v1@v1.T))
    Wv = (1/(v1.shape[1]))*(v2-Av@v1)@((v2-Av@v1).T)
    
    return Av, Wv

def fit_ofc_model(Av, Wv, delta, tau):
    """
    Fits optimal feedback control model parameters.
    
    Parameters
    ----------
    Av: np.array
        system evolution matrix, computed using fit_vel_model
    Wv: np.array
        white-noise covariance, computed using fit_vel_model
    delta: np.array
        timestep of data/model
    tau: float
        time constant, approximate duration of a movement
        
    Returns
    -------
    A, B, W, L: np.array
        OFC model parameter matrices
    
    """
    A = np.zeros((4,4))
    A[0,0] = 1
    A[0,2] = delta
    A[1,1] = 1
    A[1,3] = delta
    A[2,2] = Av[0,0]   
    A[3,3] = Av[1,1]

    B = np.zeros((4,2))
    B[2,0] = delta
    B[3,1] = delta

    W = np.zeros((4,4))
    W[2,2] = Wv[0,0]
    W[3,3] = Wv[1,1]

    # solve Riccati equation
    wx = np.identity(4) * 1.5 * (tau)**2   
    wr = np.identity(2) * (tau)**4
    wx[0,0] = 1000
    wx[1,1] = 1000
    P = linalg.solve_discrete_are(A,B,wx,wr)
    L = la.inv(wr+B.T@P@B)@B.T@P@A
    
    return A, B, W, L