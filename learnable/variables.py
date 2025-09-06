import numpy as np
from functions import compute_rate_k_torch
from scipy.stats import chi2
from scipy.io import loadmat

import torch
from scipy.stats import chi2
from scipy.io import loadmat

class GlobalConstants:
    def __init__(self, H_HAT, snr_db=10, snrest_db=[11], Nt=4, Nr=4, K=4, Pt=16, Pin=0.5, h_hat_id=-1, device='cpu'):
        self.NT = Nt
        self.NR = Nr
        self.K = K
        self.PT = Pt

        self.SNR_DB = snr_db
        self.SIGMA = torch.tensor(10 ** (-snr_db / 20), dtype=torch.float32, device=device)

        r2 = chi2.ppf(Pin, df=Nt*Nr*2)
        r2 = torch.tensor(r2, dtype=torch.float32, device=device)

        self.eps = []
        self.B = []
        self.Binv = []
        self.SIGMAEST = []
        self.SNREST_DB = snrest_db

        for k in range(K):
            sigma_est = torch.tensor(10 ** (snrest_db[k] / 20), dtype=torch.float32, device=device)
            self.SIGMAEST.append(sigma_est)
            eps_k = np.sqrt((1 + 1 / sigma_est**2) * r2)
            self.eps.append(eps_k)
            self.B.append(torch.eye(Nt*Nr, dtype=torch.cfloat, device=device)/eps_k)
            self.Binv.append(torch.eye(Nt*Nr, dtype=torch.cfloat, device=device) * eps_k)

        if h_hat_id == -1:
            H_HAT_mat = loadmat("H_HAT.mat")
            H_HAT = H_HAT_mat['H_HAT'][h_hat_id]

        # Convert H_HAT to torch tensor
        self.H_HAT = torch.tensor(H_HAT, dtype=torch.cfloat, device=device)

        # Scale each channel realization
        # for k in range(K):
        #     scale = self.SIGMAEST[k]
        #     self.H_HAT[k] *= scale

import torch

class VariablesA:
    def __init__(self, constants, delta_k_id=-1, device='cpu'):
        self.y = [torch.randn(constants.NR, 1, dtype=torch.cfloat, device=device) for _ in range(constants.K)]

        self.Delta = []
        # Load Delta_k from .mat file and convert to torch tensor
        Delta_np = loadmat("Delta_k.mat")["Delta_k"][delta_k_id] 
        if delta_k_id == -1:
            Delta_np = np.zeros_like(Delta_np)
        # Make Delta a learnable parameter
        for k in range(constants.K):
            scale = np.sqrt(1/ (1 + 10**(constants.SNREST_DB[k]/10)))
            self.Delta.append(torch.nn.Parameter(torch.tensor(Delta_np[k]*scale, dtype=torch.cfloat, device=device), requires_grad=True))
        # Each delta_k as a learnable parameter (flattened)
        self.delta = [torch.nn.Parameter(torch.tensor(Delta_k.reshape(-1, 1), dtype=torch.cfloat, device=device), requires_grad=True)
                      for Delta_k in Delta_np]
        

class VariablesB:
    def __init__(self, constants, device='cpu'):
        self.LAMB = torch.ones(constants.K, device=device)
        self.ALPHA = torch.tensor(5.0, device=device)
        self.BETA = torch.tensor(5.0, device=device)
        self.t = torch.tensor(0.0, device=device)

        # Make V_k learnable parameters
        V_init = []
        total_power = 0
        for _ in range(constants.K):
            V_k = torch.randn(constants.NT, 1, dtype=torch.cfloat, device=device)
            V_init.append(V_k)
            total_power += torch.norm(V_k)**2
        scaling_factor = torch.sqrt(torch.tensor(constants.PT, device=device) / total_power)
        V_init = [scaling_factor * v for v in V_init]
        # Store as nn.ParameterList for optimization
        self.V = torch.nn.ParameterList([torch.nn.Parameter(v, requires_grad=True) for v in V_init])

        self.W = [torch.randn(constants.NR, 1, dtype=torch.cfloat, device=device) for _ in range(constants.K)]

def initialize_t(A, B, constants):
    """
    Initialize t to be feasible (t <= sum_k g1_k).
    """

    K = constants.K
    sigma2 = constants.SIGMA**2

    total_g1 = 0
    for k in range(K):
        Nr = constants.NR
        Nt = constants.NT
        H_hat_k = constants.H_HAT[k]
        delta_k = A.delta[k].reshape(Nr, Nt)
        H_k = H_hat_k + delta_k

        v_k = B.V[k]
        w_k = B.W[k]

        # Interference
        # interference = sigma2 * np.eye(Nr, dtype=complex)
        # for n in range(K):
        #     if n != k:
        #         H_hat_n = constants.H_HAT[n]
        #         delta_n = A.delta[n].reshape(Nr, Nt)
        #         H_n = H_hat_n + delta_n
        #         v_n = B.V[n]
        #         interference += H_n @ (v_n @ v_n.conj().T) @ H_n.conj().T

        # SINR_k = np.real(w_k.conj().T @ H_k @ v_k + v_k.conj().T @ H_k.conj().T @ w_k \
        #         - w_k.conj().T @ interference @ w_k).item()

        # SINR_k = max(SINR_k, -0.99)  # clip

        # g1_k = np.log(1 + SINR_k)
        g1_k = compute_rate_k_torch(A, B, constants, k)

        delta_penalty = (delta_k.reshape(-1,1).conj().T @ constants.B[k] @ delta_k.reshape(-1,1)).real.item() - 1
        g1_k += B.LAMB[k] * delta_penalty

        total_g1 += g1_k

    # Initialize t
    B.t = 0.5 * total_g1  # for example
    print(f"Initialized t to {B.t:.6e} to ensure feasibility.")

    return B