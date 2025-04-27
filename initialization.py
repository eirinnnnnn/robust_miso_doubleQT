import numpy as np

def initialize_A(constants):
    Nt = constants["NT"]
    Nr = constants["NR"]
    K = constants["K"]

    sigma_est = constants["SIGMAEST"]

    ## Result: y_k^\star, \bDelta_k^\star    

    ## initialize a feasible \bDelta_k 

    Delta = []
    delta = []
    for _ in range(K):
        Delta.append(np.random.normal(0, 0.5*sigma_est, (Nr, Nt)) + 1j*np.random.normal(0, 0.5*sigma_est, (Nr, Nt)))
        delta.append(Delta[-1].flatten(-1,1))

    A = {
        "Delta": Delta,
        "delta": delta,
    }
    return A

def initialize_B(constants):
    Nt = constants["NT"]
    Nr = constants["NR"]
    K = constants["K"]

    Pt = constants["PT"]
    ## initialize precoders
    V = []
    for _ in range(K):
        V.append(np.random.normal(0, 0.5, (Nr, Nt)) + 1j*np.random.normal(0, 0.5, (Nr, Nt))) /  


    B = {
        "LAMB": np.array([1 for _ in range(K)]),
        "ALPHA": 0.5,
        "BETA": 0.5,
        "t": 0,
        ###
        "V": V
    }
    return B
