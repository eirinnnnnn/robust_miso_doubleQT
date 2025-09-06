import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.io import loadmat
import torch

# ----------------------------
# Utilities
# ----------------------------

def _get_V_numpy(B, A, constants):
    """
    Helper: get current precoders as NumPy arrays (evaluation only).
    """
    V_torch = B.get_V(A, constants)  # list of (Nt,1) complex torch tensors
    V_np = []
    for v in V_torch:
        if hasattr(v, "detach"):
            V_np.append(v.detach().cpu().numpy())
        else:
            V_np.append(np.asarray(v))
    return V_np

def wrap_precoders_for_test(V_np, constants):
    """
    Wraps a list of NumPy precoders (from wmmse_sum_rate or zf_precoder) into a dummy B object
    with a get_V method compatible with compute_rate_test.
    """
    class DummyB:
        def get_V(self, A, constants):
            return V_np
    return DummyB()

# ----------------------------
# Evaluation (NumPy)
# ----------------------------

def compute_rate_test(A, B, constants, Delta_k, samp=1000, outage_percentile=5):
    """
    Compute the rate R_k for each user given current A (delta) and NN precoders (B.get_V).
    Evaluation-only: all inputs are detached and converted to NumPy.
    """
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = float(constants.SIGMA**2)

    # Snapshot to NumPy to avoid autograd
    if hasattr(constants.H_HAT, "detach"):
        H_HAT = [h.detach().cpu().numpy() for h in constants.H_HAT]
    else:
        H_HAT = constants.H_HAT

    # Use network-generated precoders
    V_list = _get_V_numpy(B, A, constants)

    if hasattr(Delta_k, "detach"):
        Delta_np = Delta_k.detach().cpu().numpy()
    else:
        Delta_np = Delta_k

    rate = []
    mean = []
    var = []
    outage = []

    for k in range(constants.K):
        rate_k = []
        for s in range(samp):
            scale = np.sqrt(1.0 / (1.0 + 10.0**(constants.SNREST_DB[k] / 10.0)))
            H_hat_k = H_HAT[k]
            delta_k = scale * Delta_np[s][k]
            H_k = H_hat_k + delta_k  # effective channel

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V_list[n]
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V_list[k]
            SINR = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_val = float(np.real(SINR.item()))

            rate_k.append(np.log2(1 + SINR_val))
        rate.append(rate_k)
        mean.append(np.mean(rate_k))
        var.append(np.var(rate_k))
        outage.append(np.percentile(rate_k, outage_percentile))

    return rate, mean, var, outage


# ----------------------------
# Torch (training-time pieces)
# ----------------------------

def compute_rate_k_torch(A, B, constants, k, V_list=None):
    """
    Single-user rate with NN-generated precoders.
    If V_list is provided (list of (Nt,1) complex tensors), it will be used;
    otherwise we will call B.get_V(A, constants).
    """
    device = constants.H_HAT.device
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2

    H_hat_k = constants.H_HAT[k]
    delta_k = A.delta[k].reshape(Nr, Nt)
    H_k = H_hat_k + delta_k

    # Use provided V_list or fetch from NN
    if V_list is None:
        V_list = B.get_V(A, constants)  # list of (Nt,1)
        # print(V_list)

    I = sigma2 * torch.eye(Nr, dtype=torch.cfloat, device=device)
    interference = I.clone()
    for n in range(constants.K):
        if n != k:
            v_n = V_list[n]
            interference = interference + H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

    v_k = V_list[k]
    SINR_num = v_k.conj().T @ H_k.conj().T @ torch.linalg.pinv(interference) @ H_k @ v_k
    SINR_num = SINR_num.real.squeeze()

    rate_k = torch.log2(1 + SINR_num)
    def check_tensor(name, t):
        finite = torch.isfinite(t)
        if t.is_complex():
            abs_t = t.abs()
            print(f"[DEBUG] {name}: abs min={abs_t.min().item()}, abs max={abs_t.max().item()}, finite={finite.all().item()}")
        else:
            print(f"[DEBUG] {name}: min={t.min().item()}, max={t.max().item()}, finite={finite.all().item()}")
        if not finite.all():
            print(f"[DEBUG] Non-finite values in {name}. Shape: {t.shape}")
            print(t)
    # print("\n==== DEBUG TENSOR SUMMARY ====")
    # check_tensor("H_hat_k", H_hat_k)
    # check_tensor("delta_k", delta_k)
    # check_tensor("H_k", H_k)
    # for idx, v in enumerate(V_list):
    #     check_tensor(f"V_list[{idx}]", v)
    # print("============================\n")
    return rate_k


def update_lagrangian_variables_learnable(A, B, constants, ite, robust=True):
    """
    Update dual variables (beta and lambda) WITHOUT creating autograd history.
    Beta’s residual uses current NN precoders.
    """
    eta_0 = 1e-1  # (snr = 0, snrest>5)
    eta_beta = eta_0 
    eta_lamb = 1000.0
    K = constants.K
    Pt = constants.PT

    with torch.no_grad():
        # Current precoders (no grad)
        V_list = B.get_V(A, constants)

        # Power constraint residual and beta update
        total_power = sum(torch.linalg.norm(V_list[k])**2 for k in range(K))
        res_beta = Pt - total_power
        step_beta = eta_beta * abs(res_beta) + 1e-8
        if torch.is_tensor(B.BETA):
            new_beta = torch.clamp(B.BETA - step_beta * res_beta, min=0.0)
            B.BETA = float(new_beta)
        else:
            B.BETA = max(0.0, float(B.BETA) - float(step_beta * res_beta))

        # Lambda updates for mismatch constraints
        for k in range(K):
            if robust:
                delt = A.delta[k]
                cval = (delt.conj().T @ (constants.B[k]) @ delt).real - 1  # scalar tensor
                step_l = eta_lamb * abs(cval) + 1e-8
                lam_val = B.LAMB[k] if k < len(B.LAMB) else 0.0
                lam_val = float(lam_val) if torch.is_tensor(lam_val) else float(lam_val)
                lam_new = max(0.0, lam_val + float(step_l * cval))
                B.LAMB[k] = lam_new
            else:
                B.LAMB[k] = 0
    return B


def lossf(A, B, constants, V_list=None):
    """
    Lagrangian (we maximize it during training → usually minimize negative outside).
    Accepts optional V_list to avoid recomputing B.get_V.
    """
    if V_list is None:
        V_list = B.get_V(A, constants)

    # Sum of rates
    total_rate = 0
    for k in range(constants.K):
        total_rate = total_rate + compute_rate_k_torch(A, B, constants, k, V_list=V_list)

    # Power term
    pwr = sum(torch.norm(V_list[k])**2 for k in range(constants.K))

    # Mismatch constraints
    mismatch = 0
    for k in range(constants.K):
        delta_k = A.delta[k].reshape(-1, 1)
        mismatch = mismatch + B.LAMB[k] * ((delta_k.conj().T @ constants.B[k] @ delta_k) - 1)

    return (total_rate + B.BETA * (constants.PT - pwr) + mismatch).real


def update_loop_learnable(
    A, B, constants,
    outer_tol=1e-6, max_outer_iter=100,
    lr_model=1e-3, lr_delta=1e-3,
    robust=True,
    opt_model=None,
    opt_delta=None,
):
    """
    Outer loop for updating the NN precoder (B.model) and (optionally) A.delta.

    - Uses an optimizer for B.model.parameters() rather than in-place updates of B.V.
    - Optionally optimizes A.delta via a separate optimizer (default: SGD).
    - Updates dual variables via update_lagrangian_variables_learnable.
    """
    device = constants.H_HAT.device

    # Build default optimizers if not supplied
    if opt_model is None:
        opt_model = torch.optim.Adam(B.model.parameters(), lr=lr_model)
    if opt_delta is None:
        # A.delta is a list of nn.Parameter
        opt_delta = torch.optim.Adam(A.delta, lr=lr_delta)

    prev_outer_lagrangian = None
    lagrangian_trajectory = []
    alpha_trajectory = []
    beta_trajectory = []
    t_trajectory = []
    res1 = []
    res2 = []
    converged = False
    lr_delta = 1e-4
    for outer_iter in range(max_outer_iter):
        opt_model.zero_grad(set_to_none=True)
        opt_delta.zero_grad(set_to_none=True)

        # Forward once to get V_list from the NN
        V_list = B.get_V(A, constants)

        # Compute Lagrangian
        L = lossf(A, B, constants, V_list=V_list)


        # We maximize L → so minimize (-L)
        loss = -L
        loss.backward()

        # Step optimizers
        opt_model.step()
        if robust:
            # Flip gradients for delta to do descent
            for param in A.delta:
                if param.grad is not None:
                    param.grad.data.mul_(-1)
            opt_delta.step()
            # Clamp delta after optimizer step
            # with torch.no_grad():
            #     for k in range(constants.K):
            #         max_delta = constants.eps[k] 
            #         A.delta[k].data.real.clamp_(-max_delta, max_delta)
            #         A.delta[k].data.imag.clamp_(-max_delta, max_delta)
            #         A.Delta[k].copy_(A.delta[k].reshape(constants.NR, constants.NT))

        # Invalidate any internal V cache if present
        if hasattr(B, "_V_cache"):
            B._V_cache = None
            B._cache_inputs = None

        # Dual updates (no grad)
        B = update_lagrangian_variables_learnable(A, B, constants, outer_iter, robust)
        beta_trajectory.append(float(B.BETA))

        # Primal residuals for logging
        with torch.no_grad():
            # Power slack
            V_list_now = B.get_V(A, constants)
            slack_pwr = constants.PT - sum(torch.linalg.norm(v)**2 for v in V_list_now)
            res2.append(float(slack_pwr.detach()))

            # Mismatch residual
            msmtch = 0.0
            for k in range(constants.K):
                delta_k = A.delta[k].reshape(-1, 1)
                msmtch = msmtch + B.LAMB[k] * ( (delta_k.conj().T @ constants.B[k] @ delta_k) - 1 ).real
            msmtch = float(msmtch)
            res1.append(msmtch)

            current_outer_lagrangian = float(L.detach().item())
            lagrangian_trajectory.append(current_outer_lagrangian)

        # Convergence check
        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)

            print(f"slk_pwr = {float(slack_pwr):.6e}, beta = {float(B.BETA):.6e}, lambda = {float(torch.mean(B.LAMB))}, msmtch = {float(msmtch):.6e}")
            print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}")

            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2


# ----------------------------
# Baselines (unchanged)
# ----------------------------

def wmmse_sum_rate(constants, max_iter=100, tol=1e-4):
    """
    Classic WMMSE algorithm for sum-rate maximization with perfect CSI.
    Returns V as a list of (Nt,1) complex numpy arrays.
    """
    if hasattr(constants.H_HAT, "detach"):
        H_HAT = constants.H_HAT.detach().cpu().numpy()
    else:
        H_HAT = np.array(constants.H_HAT)
    Pt = float(constants.PT)
    sigma2 = float(constants.SIGMA**2)
    K, Nr, Nt = H_HAT.shape

    V = []
    total_power = 0
    for _ in range(K):
        v_k = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)
        V.append(v_k)
        total_power += np.linalg.norm(v_k)**2
    scaling = np.sqrt(Pt / total_power)
    V = [scaling * v_k for v_k in V]

    for it in range(max_iter):
        # Step 1: Update receive filters W
        W = []
        for k in range(K):
            Hk = H_HAT[k]
            interf = sigma2 * np.eye(Nr, dtype=complex)
            for j in range(K):
                if j != k:
                    interf += Hk @ V[j] @ V[j].conj().T @ Hk.conj().T
            v_k = V[k]
            W_k = np.linalg.inv(interf + Hk @ v_k @ v_k.conj().T @ Hk.conj().T) @ Hk @ v_k
            W.append(W_k)

        # Step 2: Update MSE weights U
        U = []
        for k in range(K):
            Hk = H_HAT[k]
            v_k = V[k]
            W_k = W[k]
            e_k = 1 - 2 * np.real(W_k.conj().T @ Hk @ v_k) + \
                  W_k.conj().T @ (sigma2 * np.eye(Nr) + sum(Hk @ V[j] @ V[j].conj().T @ Hk.conj().T for j in range(K))) @ W_k
            U_k = 1.0 / np.real(e_k)
            U.append(U_k)

        # Step 3: Update transmit precoders V
        V_new = []
        total_power = 0
        for k in range(K):
            Hk = H_HAT[k]
            W_k = W[k]
            U_k = U[k]
            A = np.zeros((Nt, Nt), dtype=complex)
            b = np.zeros((Nt, 1), dtype=complex)
            for j in range(K):
                Hj = H_HAT[j]
                W_j = W[j]
                U_j = U[j]
                A += U_j * Hj.conj().T @ W_j @ W_j.conj().T @ Hj
                if j == k:
                    b += U_j * Hj.conj().T @ W_j
            A += 1e-8 * np.eye(Nt)
            v_k = np.linalg.solve(A, b)
            V_new.append(v_k)
            total_power += np.linalg.norm(v_k)**2

        scaling = np.sqrt(Pt / total_power)
        V_new = [scaling * v_k for v_k in V_new]

        norm_diffs = [float(np.linalg.norm(V[k] - V_new[k])) for k in range(K)]
        V = V_new
        if max(norm_diffs) < tol:
            break

    return V


def zf_precoder(constants):
    """
    Zero-Forcing (NumPy) baseline.
    """
    if hasattr(constants.H_HAT, "detach"):
        H_HAT = constants.H_HAT.detach().cpu().numpy()
    else:
        H_HAT = np.array(constants.H_HAT)
    Pt = float(constants.PT)
    K, Nr, Nt = H_HAT.shape

    H_stack = np.vstack([H_HAT[k] for k in range(K)])  # (K*Nr, Nt)
    H_pinv = np.linalg.pinv(H_stack)                   # (Nt, K*Nr)
    V_zf = H_pinv.T                                    # (K*Nr, Nt)

    V = []
    total_power = 0
    for k in range(K):
        v_k = V_zf[k*Nr:(k+1)*Nr, :].T  # (Nt, Nr)
        v_k = v_k[:, 0].reshape(Nt, 1)  # single stream
        V.append(v_k)
        total_power += np.linalg.norm(v_k)**2

    scaling = np.sqrt(Pt / total_power)
    V = [scaling * v_k for v_k in V]
    return V
