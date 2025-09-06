
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

import numpy as np

# Reuse user's helpers
from functions import compute_rate_k_torch
from variables import GlobalConstants, VariablesA, VariablesB

# ---------- Utilities ----------

def _concat_features_per_user(H_hat_k: torch.Tensor,
                              delta_k_mat: torch.Tensor,
                              v_prev_k: torch.Tensor) -> torch.Tensor:
    """
    Build a real-valued feature vector per user by concatenating:
      [Re(H), Im(H), Re(Delta), Im(Delta), Re(v_prev), Im(v_prev)]
    Shapes:
      H_hat_k: (Nr, Nt) complex
      delta_k_mat: (Nr, Nt) complex
      v_prev_k: (Nt, 1) complex
    Returns:
      feat: (F,) real tensor
    """
    def _ri(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x.real.flatten(), x.imag.flatten()], dim=0)

    feat = torch.cat([_ri(H_hat_k), _ri(delta_k_mat), _ri(v_prev_k)], dim=0).float()
    return feat


def _power_normalize(V_list: List[torch.Tensor], Pt: float) -> List[torch.Tensor]:
    """
    Scale list of complex precoders so that total transmit power equals Pt.
    V_k shape: (Nt, 1) complex.
    """
    with torch.no_grad():
        total_pwr = sum(torch.norm(v)**2 for v in V_list)
        total_pwr = torch.clamp(total_pwr.real, min=1e-12)
        scale = torch.sqrt(torch.tensor(Pt, dtype=V_list[0].dtype) / total_pwr)
    return [v * scale for v in V_list]


# ---------- Model ----------

class PrecoderMLP(nn.Module):
    """
    Per-user MLP that maps [H_hat, Delta, v_prev] -> v_k (complex, Nt x 1).
    Implemented as two heads for real/imag parts.
    """
    def __init__(self, Nr: int, Nt: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        feat_dim = 2 * (Nr * Nt + Nr * Nt + Nt)  # Re/Im for H, Delta, v_prev
        layers = []
        in_dim = feat_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 2 * Nt))  # output real+imag for Nt entries
        self.net = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        feats: (K, F) real
        returns: complex tensor of shape (K, Nt, 1)
        """
        out = self.net(feats)  # (K, 2Nt)
        K, twoNt = out.shape
        Nt = twoNt // 2
        out_real, out_imag = out[:, :Nt], out[:, Nt:]
        v = torch.complex(out_real, out_imag).unsqueeze(-1)  # (K, Nt, 1)
        return v


# ---------- Training Loop ----------

@torch.no_grad()
def _assign_precoders_to_B(B: VariablesB, V_list: List[torch.Tensor]) -> None:
    # Replace B.V content with learned V_list as nn.Parameters
    out = []
    for v in V_list:
        out.append(nn.Parameter(v.clone(), requires_grad=True))
    B.V = nn.ParameterList(out)


def update_loop_learnable_nn(A: VariablesA,
                             B: VariablesB,
                             constants: GlobalConstants,
                             max_outer_iter: int = 8000,
                             outer_tol: float = 1e-5,
                             robust: bool = True,
                             lr: float = 1e-3,
                             hidden: int = 512,
                             depth: int = 3,
                             device: str = "cpu"
                             ) -> Tuple[VariablesB, list, list, list, list, list, list]:
    """
    Neural precoder (MLP) for B.V with the SAME loss as user's current one.
    Returns the same tuple signature as update_loop_learnable().
    """
    K = constants.K
    Nr = constants.NR
    Nt = constants.NT
    Pt = float(constants.PT)

    model = PrecoderMLP(Nr, Nt, hidden=hidden, depth=depth).to(device)

    # Optimizer will update model params (+ optionally A.delta if robust)
    params = list(model.parameters())
    if robust:
        params += [p for p in A.delta]  # learn mismatch too (as in user's loop)
    opt = torch.optim.Adam(params, lr=lr)

    # Trajectory buffers (match user's return signature lengths/meanings)
    lagrangian_trajectory = []
    alpha_trajectory = []     # kept for signature; not used
    beta_trajectory = []
    t_trajectory = []         # kept for signature; not used
    res1_trajectory = []      # mismatch residuals
    res2_trajectory = []      # power residuals

    prev_outer = None
    converged = False

    # Initialize "previous V" with B.V
    V_prev = [v.detach().clone().to(device) for v in B.V]

    for it in range(max_outer_iter):
        opt.zero_grad(set_to_none=True)

        # ---- Build features per user ----
        feats = []
        for k in range(K):
            Hk = constants.H_HAT[k].to(device)
            Dk = A.delta[k].reshape(Nr, Nt).to(device)
            vk_prev = V_prev[k].to(device)
            feats.append(_concat_features_per_user(Hk, Dk, vk_prev))
        feats = torch.stack(feats, dim=0)  # (K, F)

        # ---- Forward NN -> V (K, Nt, 1) ----
        v_batch = model(feats)  # complex
        V_list = [v_batch[k] for k in range(K)]

        # Power normalize
        V_list = _power_normalize(V_list, Pt)

        # ---- Compute loss (same as user's) ----
        # sum_k rate_k(A, V_list) + beta*(PT - pwr) + sum_k lambda_k * (delta^H B delta - 1)
        sum_rate = 0.0
        for k in range(K):
            # Temporarily attach the V_list to B for compute_rate_k_torch convenience
            # but avoid replacing B.V permanently mid-training
            # We'll emulate by a tiny shim object
            pass
        # Instead, evaluate rate directly using the same routine but a local copy of B
        B_local = VariablesB(constants, device=device)
        # overwrite with current V_list (no grad params inside B_local)
        with torch.no_grad():
            for k in range(K):
                B_local.V[k].data.copy_(V_list[k])

        # Sum-rate
        total_rate = 0.0
        for k in range(K):
            rk = compute_rate_k_torch(A, B_local, constants, k)
            total_rate = total_rate + rk

        # Power term
        total_pwr = sum(torch.norm(v)**2 for v in V_list)
        power_residual = torch.tensor(Pt, dtype=total_pwr.dtype, device=total_pwr.device) - total_pwr.real

        # Mismatch constraints
        mismatch = 0.0
        for k in range(K):
            if robust:
                delt = A.delta[k]
                mismatch = mismatch + B.LAMB[k] * ((delt.conj().T @ constants.B[k] @ delt).real - 1.0)
            else:
                mismatch = mismatch + 0.0

        lagrangian = (total_rate + B.BETA * power_residual + mismatch).real.squeeze()

        # ---- Backprop + step ----
        (-lagrangian).backward()  # maximize lagrangian -> minimize negative
        opt.step()

        # ---- Dual updates (beta, lambda) without autograd history ----
        with torch.no_grad():
            # update beta (projected gradient ascent on dual)
            eta = 1e-1
            step_beta = eta * abs(power_residual) + 1e-8
            new_beta = torch.clamp(torch.tensor(float(B.BETA)) - step_beta * power_residual, min=0.0)
            B.BETA = float(new_beta)

            # lambda updates
            for k in range(K):
                if robust:
                    cval = (A.delta[k].conj().T @ constants.B[k] @ A.delta[k]).real - 1.0
                    step_l = eta * abs(cval) + 1e-8
                    lam_new = max(0.0, float(B.LAMB[k]) + float(step_l * cval))
                    B.LAMB[k] = lam_new
                else:
                    B.LAMB[k] = 0.0

        # ---- Logging ----
        lagrangian_trajectory.append(float(lagrangian.detach().cpu()))
        beta_trajectory.append(float(B.BETA))
        res2_trajectory.append(float(power_residual.detach().cpu()))
        if robust:
            msmtch = 0.0
            for k in range(K):
                msmtch += float(((A.delta[k].conj().T @ constants.B[k] @ A.delta[k]).real - 1.0).detach().cpu())
            res1_trajectory.append(msmtch)
        else:
            res1_trajectory.append(0.0)

        # Convergence check
        if prev_outer is not None:
            delta = abs(lagrangian_trajectory[-1] - prev_outer)
            if delta < outer_tol:
                converged = True
                # finalize learned V into B
                _assign_precoders_to_B(B, _power_normalize(V_list, Pt))
                break
        prev_outer = lagrangian_trajectory[-1]

        # Prepare V_prev for the next iteration (detach to avoid long graphs)
        V_prev = [v.detach().clone() for v in V_list]

    if not converged:
        # finalize learned V into B
        _assign_precoders_to_B(B, _power_normalize(V_prev, Pt))

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1_trajectory, res2_trajectory
