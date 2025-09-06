

import numpy as np
import sys
import matplotlib.pyplot as plt

from scipy.io import loadmat
import torch

def compute_rate_test(A, B, constants, Delta_k, snr ,samp=1000, outage_percentile=5):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
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

    if hasattr(B.V[0], "detach"):
        V_list = [v.detach().cpu().numpy() for v in B.V]
    else:
        V_list = B.V

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
            # scale = 1.0
            scale = np.sqrt(1/ (1 + 10**(snr/10)))
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

def compute_rate_k_torch(A, B, constants, k):
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2

    H_hat_k = (constants.H_HAT[k])
    delta_k = A.delta[k].reshape(Nr, Nt) 
    H_k = H_hat_k + delta_k

    V = B.V 
    # V = [v for v in B.V]  

    interference = sigma2 * torch.eye(Nr, dtype=torch.cfloat)
    for n in range(constants.K):
        if n != k:
            v_n = V[n]
            interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

    v_k = V[k]
    SINR_numerator = v_k.conj().T @ H_k.conj().T @ torch.linalg.pinv(interference) @ H_k @ v_k
    SINR_numerator = SINR_numerator.real.squeeze()

    rate_k = torch.log2(1 + SINR_numerator)
    return rate_k


def update_lagrangian_variables_learnable(A, B, constants, ite, robust=True):
    """
    Update dual variables (beta and lambda) WITHOUT creating autograd history.
    Store them as plain floats so they are treated as constants in the loss.
    """
    eta_0 = 1e-1  # (snr = 0, snrest>5)
    eta_beta = eta_0
    K = constants.K
    Pt = constants.PT

    with torch.no_grad():
        # Power constraint residual and beta update
        total_power = sum(torch.linalg.norm(B.V[k])**2 for k in range(K))
        res_beta = Pt - total_power
        step_beta = eta_beta * abs(res_beta) + 1e-8
        if torch.is_tensor(B.BETA):
            new_beta = torch.clamp(B.BETA - step_beta * res_beta, min=0.0)
            B.BETA = float(new_beta)
        else:
            B.BETA = max(0.0, float(B.BETA) - float(step_beta * res_beta))

        # Lambda updates for mismatch constraints
        for k in range(constants.K):
            if robust:
                delt = A.delta[k]
                cval = (delt.conj().T @ (constants.B[k]) @ delt).real - 1  # scalar tensor
                step_l = eta_beta * abs(cval) + 1e-8
                lam_val = B.LAMB[k] if k < len(B.LAMB) else 0.0
                lam_val = float(lam_val) if torch.is_tensor(lam_val) else float(lam_val)
                lam_new = max(0.0, lam_val + float(step_l * cval))
                B.LAMB[k] = lam_new
            else:
                B.LAMB[k] = 0
    return B


def lossf(A, B, constants):
    pwr = 0
    loss = 0

    for k in range(constants.K): 
    # sum rate
        loss += compute_rate_k_torch(A, B, constants, k)
    # mismatch constraints

    # power constraint
    pwr = sum(torch.norm(B.V[k])**2 for k in range(constants.K))
    mismatch = sum(B.LAMB[k] * (A.delta[k].conj().T @ (
        constants.B[k]) 
        @ A.delta[k]-1) for k in range(constants.K))


    return (loss + B.BETA * (constants.PT - pwr) + mismatch).real

def update_loop_learnable(A, B, constants, 
                         outer_tol=1e-6, max_outer_iter=100, 
                         inner_tol=1e-4, max_inner_iter=500,
                         robust = True):
    """
    Outer loop for updating B (V, Lambda, t, alpha, beta).
    Logs alpha and beta values to inspect KKT conditions.
    """

    K = constants.K
    Nr = constants.NR
    Pt = constants.PT

    prev_outer_lagrangian = None
    lagrangian_trajectory = []
    alpha_trajectory = []  
    beta_trajectory = []   
    t_trajectory = []   
    res1 = []
    res2 = []
    converged = False
    lr_delta = 1e-4  
    lr_v = 1e-4 
    for outer_iter in range(max_outer_iter):

        for k in range(constants.K):
            if A.delta[k].grad is not None:
                A.delta[k].grad = None
            if B.V[k].grad is not None:
                B.V[k].grad = None

        # Compute loss
        loss = lossf(A, B, constants)
        loss.backward()

        # Parameter updates without creating new graph links
        with torch.no_grad():
            if robust:
                for k in range(constants.K):
                    A.delta[k].add_(-lr_delta * A.delta[k].grad)
                    A.Delta[k].copy_(A.delta[k].reshape(constants.NR, constants.NT))
            for k in range(constants.K):
                B.V[k].add_(lr_v * B.V[k].grad)

        B = update_lagrangian_variables_learnable(A, B, constants, outer_iter, robust)
        beta_trajectory.append(float(B.BETA))    

        slack_pwr =  constants.PT - sum(torch.linalg.norm(B.V[k]) ** 2 for k in range(constants.K)) 
        res2.append(float(slack_pwr.detach() if torch.is_tensor(slack_pwr) else slack_pwr))

        msmtch =  sum(B.LAMB[k]*(A.delta[k].conj().T@constants.B[k]@A.delta[k] -1).real for k in range(constants.K)) 
        res1.append(float(msmtch) if torch.is_tensor(msmtch) else msmtch)


        # current_outer_lagrangian = lossf(A, B, constants)
        current_outer_lagrangian = float(loss.detach().real.item() if torch.is_tensor(loss) else loss)
        lagrangian_trajectory.append(current_outer_lagrangian)

        # === Check convergence ===
        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}")
            # print(f"slk_pwr = {float(slack_pwr):.6e}, beta = {float(B.BETA):.6e}, msmtch = {float(msmtch):.6e}")

            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}, alpha = {B.ALPHA}, beta = {B.BETA}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2


# def wmmse_sum_rate(constants, max_iter=100, tol=1e-4):
#     """
#     Classic WMMSE algorithm for sum-rate maximization with perfect CSI.
#     Args:
#         H_HAT: list or array of shape (K, Nr, Nt), estimated channel for each user
#         Pt: total transmit power
#         sigma2: noise variance
#         max_iter: maximum number of iterations
#         tol: convergence tolerance
#     Returns:
#         V: list of (Nt, 1) precoders for each user
#         sum_rate: achieved sum rate (bps/Hz)
#     """
#     # Convert all relevant variables to NumPy
#     if hasattr(constants.H_HAT, "detach"):
#         H_HAT = constants.H_HAT.detach().cpu().numpy()
#     else:
#         H_HAT = np.array(constants.H_HAT)
#     Pt = float(constants.PT)
#     sigma2 = float(constants.SIGMA**2)
#     K, Nr, Nt = H_HAT.shape

#     # Initialize V randomly and normalize to meet power constraint
#     V = []
#     total_power = 0
#     for _ in range(K):
#         v_k = np.random.randn(Nt, 1) + 1j * np.random.randn(Nt, 1)
#         V.append(v_k)
#         total_power += np.linalg.norm(v_k)**2
#     scaling = np.sqrt(Pt / total_power)
#     V = [scaling * v_k for v_k in V]

#     for it in range(max_iter):
#         # Step 1: Update receive filters W
#         W = []
#         for k in range(K):
#             Hk = H_HAT[k]
#             interf = sigma2 * np.eye(Nr, dtype=complex)
#             for j in range(K):
#                 if j != k:
#                     interf += Hk @ V[j] @ V[j].conj().T @ Hk.conj().T
#             v_k = V[k]
#             W_k = np.linalg.inv(interf + Hk @ v_k @ v_k.conj().T @ Hk.conj().T) @ Hk @ v_k
#             W.append(W_k)

#         # Step 2: Update MSE weights U
#         U = []
#         for k in range(K):
#             Hk = H_HAT[k]
#             v_k = V[k]
#             W_k = W[k]
#             e_k = 1 - 2 * np.real(W_k.conj().T @ Hk @ v_k) + \
#                   W_k.conj().T @ (sigma2 * np.eye(Nr) + sum(Hk @ V[j] @ V[j].conj().T @ Hk.conj().T for j in range(K))) @ W_k
#             U_k = 1.0 / np.real(e_k)
#             U.append(U_k)

#         # Step 3: Update transmit precoders V
#         V_new = []
#         total_power = 0
#         for k in range(K):
#             Hk = H_HAT[k]
#             W_k = W[k]
#             U_k = U[k]
#             A = np.zeros((Nt, Nt), dtype=complex)
#             b = np.zeros((Nt, 1), dtype=complex)
#             for j in range(K):
#                 Hj = H_HAT[j]
#                 W_j = W[j]
#                 U_j = U[j]
#                 A += U_j * Hj.conj().T @ W_j @ W_j.conj().T @ Hj
#                 if j == k:
#                     b += U_j * Hj.conj().T @ W_j
#             # Regularization for numerical stability
#             A += 1e-8 * np.eye(Nt)
#             v_k = np.linalg.solve(A, b)
#             V_new.append(v_k)
#             total_power += np.linalg.norm(v_k)**2

#         # Normalize to meet power constraint
#         scaling = np.sqrt(Pt / total_power)
#         V_new = [scaling * v_k for v_k in V_new]

#         # Check convergence
#         norm_diffs = [float(np.linalg.norm(V[k] - V_new[k])) for k in range(K)]
#         if max(norm_diffs) < tol:
#             break
#         V = V_new

#     # Compute sum rate (optional, currently commented out)
#     # sum_rate = 0
#     # for k in range(K):
#     #     Hk = H_HAT[k]
#     #     v_k = V[k]
#     #     interf = sigma2 * np.eye(Nr, dtype=complex)
#     #     for j in range(K):
#     #         if j != k:
#     #             interf += Hk @ V[j] @ V[j].conj().T @ Hk.conj().T
#     #     signal = v_k.conj().T @ Hk.conj().T @ np.linalg.inv(interf) @ Hk @ v_k
#     #     sum_rate += np.log2(1 + np.real(signal.item()))
#     return V
#     # return V, sum_rate

def wmmse_sum_rate(constants, max_iter=100, tol=1e-3):
    """
    Classic WMMSE algorithm for sum-rate maximization with perfect CSI.
    Args:
        H_HAT: list or array of shape (K, Nr, Nt), estimated channel for each user
        Pt: total transmit power
        sigma2: noise variance
        max_iter: maximum number of iterations
        tol: convergence tolerance
    Returns:
        V: list of (Nt, 1) precoders for each user
        sum_rate: achieved sum rate (bps/Hz)
    """
    # Convert all relevant variables to NumPy
    if hasattr(constants.H_HAT, "detach"):
        H_HAT = constants.H_HAT.detach().cpu().numpy()
    else:
        H_HAT = np.array(constants.H_HAT)
    Pt = float(constants.PT)
    sigma2 = float(constants.SIGMA**2)
    K, Nr, Nt = H_HAT.shape
    # Initialize V randomly and normalize to meet power constraint
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
            # Regularization for numerical stability
            A += 1e-8 * np.eye(Nt)
            v_k = np.linalg.solve(A, b)
            V_new.append(v_k)
            total_power += np.linalg.norm(v_k)**2

        # Normalize to meet power constraint
        scaling = np.sqrt(Pt / total_power)
        V_new = [scaling * v_k for v_k in V_new]

        # Check convergence
        norm_diffs = [float(np.linalg.norm(V[k] - V_new[k])) for k in range(K)]
        V = V_new
        if max(norm_diffs) < tol:
            break

    # Compute sum rate
    sum_rate = 0
    # for k in range(K):
    #     Hk = H_HAT[k]
    #     v_k = V[k]
    #     interf = sigma2 * np.eye(Nr, dtype=complex)
    #     for j in range(K):
    #         if j != k:
    #             interf += Hk @ V[j] @ V[j].conj().T @ Hk.conj().T
    #     signal = v_k.conj().T @ Hk.conj().T @ np.linalg.inv(interf) @ Hk @ v_k
    #     sum_rate += np.log2(1 + np.real(signal.item()))
    return V
    # return V, sum_rate

def zf_precoder(constants):
    """
    Zero-Forcing precoder design for multi-user MIMO.
    Args:
        constants: GlobalConstants object with H_HAT, PT, etc.
    Returns:
        V: list of (Nt, 1) precoders for each user
    """
    # Convert H_HAT to NumPy if needed
    if hasattr(constants.H_HAT, "detach"):
        H_HAT = constants.H_HAT.detach().cpu().numpy()
    else:
        H_HAT = np.array(constants.H_HAT)
    Pt = float(constants.PT)
    K, Nr, Nt = H_HAT.shape

    # Stack all user channels vertically: H = [H_1; H_2; ...; H_K] (K*Nr x Nt)
    H_stack = np.vstack([H_HAT[k] for k in range(K)])  # shape: (K*Nr, Nt)

    # ZF precoder: V = H^H (H H^H)^{-1} S
    # For MU-MISO, ZF is usually: V = H^H (H H^H)^{-1} * sqrt(Pt/K)
    # For MU-MIMO, we design v_k for each user as the k-th column of V_zf

    # Compute pseudo-inverse
    H_pinv = np.linalg.pinv(H_stack)  # shape: (Nt, K*Nr)

    # Each user's precoder is the corresponding column of H_pinv^H
    V_zf = H_pinv.T  # shape: (K*Nr, Nt)

    # Split V_zf into K blocks, each (Nr, Nt), then take the first column for each user
    V = []
    total_power = 0
    for k in range(K):
        # For each user, take the corresponding block and use the first column
        v_k = V_zf[k*Nr:(k+1)*Nr, :].T  # shape: (Nt, Nr)
        # For single-stream, use the first column
        v_k = v_k[:, 0].reshape(Nt, 1)
        V.append(v_k)
        total_power += np.linalg.norm(v_k)**2

    # Normalize to meet total power constraint
    scaling = np.sqrt(Pt / total_power)
    V = [scaling * v_k for v_k in V]

    return V