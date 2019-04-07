import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-x))
    
    return s

def d_sigmoid(x):
    s = 1/(1+np.exp(-x))
    
    return s*(1-s)

def relu(x):
    r = np.maximum(0,x)
    
    assert(x.shape == r.shape)
    
    return r

def d_relu(x):
    r = np.where(x > 0, 1, 0)
    
    assert(x.shape == r.shape)
    
    return r

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    d = tanh(x)
    return 1 - d*d

def linear(x):
    return x

def d_linear(x):
    return np.ones(shape=x.shape)

activation_functions = {"sigmoid" : (sigmoid, d_sigmoid) , "relu" : (relu, d_relu), "tanh" : (tanh, d_tanh), "linear" : (linear, d_linear)}