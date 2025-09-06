import numpy as np
from functions import compute_rate_k_torch
from scipy.stats import chi2
from scipy.io import loadmat

import torch
from scipy.stats import chi2
from scipy.io import loadmat

# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def complex_cat_real_imag(x: torch.Tensor):
    # x: (...,) complex -> (..., 2), stacking [Re, Im]
    return torch.view_as_real(x).reshape(*x.shape, 2).movedim(-1, -2).reshape(x.shape[:-1] + (2*x.shape[-1],))

def project_sum_power(V: torch.Tensor, Pt: float, eps: float = 1e-12):
    """
    V: (K, Nt) complex precoders (one stream/user).
    Enforce sum_k ||v_k||^2 = Pt via scaling.
    """
    power = torch.sum(torch.sum((V.real**2 + V.imag**2), dim=1))
    scale = torch.sqrt(torch.tensor(Pt, device=V.device) / (power + eps))
    return V * scale

class MLPPrecoder(nn.Module):
    """
    Joint MLP that looks at all users' channels and emits all users' precoders.
    Input (per batch=1): concat_k [Re(H_hat_k), Im(H_hat_k)] (+ optional side info)
    Output: V of shape (K, Nt) complex (single stream/user as in your current code).
    """
    def __init__(self, K: int, Nr: int, Nt: int, hidden=512, n_layers=3, use_eps=True):
        super().__init__()
        self.K, self.Nr, self.Nt, self.use_eps = K, Nr, Nt, use_eps

        in_dim = 2 * K * Nr * Nt                    # real-imag of all H_hat stacked
        if use_eps:
            in_dim += K                             # add K scalars of uncertainty (eps_k) as features

        out_dim = 2 * K * Nt                        # real-imag for all V_k stacked

        layers = []
        dims = [in_dim] + [hidden]* (n_layers-1) + [out_dim]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, H_HAT: torch.Tensor, eps_list=None, Pt: float = 1.0):
        """
        H_HAT: (K, Nr, Nt) complex
        eps_list: list[Tensor] length K of scalar eps_k (float32), if use_eps==True
        Returns: V (K, Nt) complex, power-projected to Pt.
        """
        K, Nr, Nt = self.K, self.Nr, self.Nt
        assert H_HAT.shape == (K, Nr, Nt)

        # Build input feature vector
        feats = []
        for k in range(K):
            hk = H_HAT[k].reshape(Nr, Nt)           # (Nr, Nt) complex
            feats.append(hk.real.reshape(-1))
            feats.append(hk.imag.reshape(-1))
        x = torch.cat(feats, dim=0)                 # (2*K*Nr*Nt,)

        if self.use_eps:
            assert eps_list is not None and len(eps_list) == K
            eps_vec = torch.stack([e.float().reshape(()) for e in eps_list])  # (K,)
            x = torch.cat([x, eps_vec], dim=0)      # (2*K*Nr*Nt + K,)

        x = x.unsqueeze(0)                          # batch=1
        y = self.net(x).squeeze(0)                  # (2*K*Nt,)

        # Add tanh activation for output stability
        y = torch.tanh(y)

        # Parse output to complex V
        V = []
        offset = 0
        for _ in range(K):
            vr = y[offset:offset+Nt]; offset += Nt
            vi = y[offset:offset+Nt]; offset += Nt
            V.append(torch.complex(vr, vi))         # (Nt,)
        V = torch.stack(V, dim=0)                   # (K, Nt) complex

        # Sum-power projection
        V = project_sum_power(V, Pt=Pt)
        return V


class GlobalConstants:
    def __init__(self, H_HAT, snr_db=10, snrest_db=[11], Nt=4, Nr=4, K=4, Pt=16, Pin=0.9, h_hat_id=-1, device='cpu'):
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
            eps_k = (1 + 1 / sigma_est**2) / r2
            self.eps.append(eps_k)
            self.B.append(eps_k * torch.eye(Nt*Nr, dtype=torch.cfloat, device=device))
            self.Binv.append(torch.eye(Nt*Nr, dtype=torch.cfloat, device=device) / eps_k)

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
        self.BETA  = torch.tensor(5.0, device=device)
        self.t     = torch.tensor(0.0, device=device)

        # --- Replace free Parameters with an NN ---
        self.model = MLPPrecoder(
            K=constants.K, Nr=constants.NR, Nt=constants.NT,
            hidden=512, n_layers=3, use_eps=True
        ).to(device)

        # Optional: initialize weights with something stable
        for m in self.model.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)

        # --- Custom initialization for output layer bias to match classic random precoders ---
        self._init_precoders(constants, device)

        # Keep W as in your current code (receive combiners if used)
        self.W = [torch.randn(constants.NR, 1, dtype=torch.cfloat, device=device) for _ in range(constants.K)]

        # Cache for latest forward (avoid recompute within a step)
        self._V_cache = None
        self._cache_inputs = None

    def _init_precoders(self, constants, device):
        """
        Initialize the last layer bias so that the initial output of the MLP matches classic random precoders,
        normalized to meet the sum power constraint.
        """
        K, Nt = constants.K, constants.NT
        # Generate classic random precoders
        V_init = []
        total_power = 0
        for _ in range(K):
            v_k = torch.randn(Nt, 1, dtype=torch.cfloat, device=device)
            V_init.append(v_k)
            total_power += torch.norm(v_k)**2
        scaling_factor = torch.sqrt(torch.tensor(constants.PT, device=device) / total_power)
        V_init = [scaling_factor * v for v in V_init]

        # Flatten real/imag parts for all users
        v_flat = []
        for v in V_init:
            v_flat.append(v.real.view(-1))
            v_flat.append(v.imag.view(-1))
        v_flat = torch.cat(v_flat, dim=0)  # shape (2*K*Nt,)

        # Set the bias of the last layer to v_flat
        last_layer = None
        for m in reversed(list(self.model.net.modules())):
            if isinstance(m, torch.nn.Linear):
                last_layer = m
                break
        if last_layer is not None:
            with torch.no_grad():
                last_layer.bias.copy_(v_flat)

    def get_V(self, A, constants):
        """
        Compute and cache V given current constants (H_HAT, Pt, eps) and (optionally) A.Delta if you want.
        Currently uses only H_HAT and eps (uncertainty radii) to produce V.
        """
        # You can also pass (H_HAT + delta_k) here if you want the network to see mismatch:
        H_in = constants.H_HAT  # shape (K, Nr, Nt) complex
        eps_list = constants.eps  # list of K tensors

        cache_key = (id(H_in), id(constants), id(self.model))
        if self._V_cache is not None and self._cache_inputs == cache_key:
            return self._V_cache

        V_mat = self.model(H_in, eps_list=eps_list, Pt=float(constants.PT))  # (K, Nt) complex
        self._V_cache = [V_mat[k].reshape(constants.NT, 1) for k in range(constants.K)]  # list of (Nt,1)
        self._cache_inputs = cache_key
        return self._V_cache



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

        v_k = B.get_V(A, constants)[k]
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