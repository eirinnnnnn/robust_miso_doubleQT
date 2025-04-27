import numpy as np
import matplotlib.pyplot as plt
def update_A(A, B, constants):
    """
    Update A (Delta, delta, y) given fixed B and constants.
    """
    K = constants.K
    delta_old = [d.copy() for d in A.delta]
    y_old = [y.copy() for y in A.y]

    for k in range(K):
        # === Update y_k given current delta_k ===
        A.y[k] = update_y_k(A.delta[k], B, constants, k)

        # === Update delta_k given updated y_k ===
        A.delta[k] = update_delta_k(A.y[k], A.delta[k], B, constants, k)

    # === Metric: sum of norm changes ===
    metric = 0
    for k in range(K):
        metric += np.linalg.norm(A.delta[k] - delta_old[k])**2
        metric += np.linalg.norm(A.y[k] - y_old[k])**2
    metric = np.sqrt(metric)

    return A, metric
def update_A_loop(A, B, constants, inner_tol=1e-3, max_inner_iter=1000, plot_lagrangian=True):
    """
    Inner loop: Alternately update y_k and delta_k for all users
    until Lagrangian convergence or maximum iterations.
    """
    K = constants.K

    prev_lagrangian = None
    lag_arr = []
    flag = None
    for inner_iter in range(max_inner_iter):
        for k in range(K):
            # === Update y_k given current delta_k ===
            A.y[k] = update_y_k(A.delta[k], B, constants, k)

            # === Update delta_k given updated y_k ===
            A.delta[k] = update_delta_k(A.y[k], A.delta[k], B, constants, k)

        # === Compute Lagrangian ===
        current_lagrangian = compute_lagrangian_A(A, B, constants)
        lag_arr.append(current_lagrangian)
        print(f"[Inner iter {inner_iter+1}] Lagrangian = {current_lagrangian:.6e}")

        if prev_lagrangian is not None:
            delta_lagrangian = abs(current_lagrangian - prev_lagrangian)
            print(f"    Lagrangian change = {delta_lagrangian:.6e}")
            if delta_lagrangian < inner_tol:
                print(f"✅ Inner loop converged at iteration {inner_iter+1}.")
                flag = True
                break

        prev_lagrangian = current_lagrangian

    else:
        print(f"⚠️ Inner loop reached max iterations without full convergence.")
        flag = False

    if plot_lagrangian:
        plt.figure()
        plt.plot(range(1, len(lag_arr)+1), lag_arr, marker='o')
        plt.xlabel('Inner Iteration')
        plt.ylabel('Total Lagrangian Value')
        plt.title('Lagrangian Trajectory in Algorithm 1')
        plt.grid(True)
        plt.savefig("L1.png")
    return A, flag

# def update_A_loop(A, B, constants, inner_tol=1e-3, max_inner_iter=50):
#     """
#     Inner loop: Update A (Delta, delta, y) given fixed B and constants.
#     Alternate updates until convergence.
#     """
#     K = constants.K

#     for inner_iter in range(max_inner_iter):
#         delta_old = [d.copy() for d in A.delta]
#         y_old = [y.copy() for y in A.y]

#         for k in range(K):
#             # === Update y_k given current delta_k ===
#             A.y[k] = update_y_k(A.delta[k], B, constants, k)

#             # === Update delta_k given updated y_k ===
#             A.delta[k] = update_delta_k(A.y[k], A.delta[k], B, constants, k)

#         # === Metric: sum of norm changes ===
#         metric = 0
#         for k in range(K):
#             metric += np.linalg.norm(A.delta[k] - delta_old[k])**2
#             metric += np.linalg.norm(A.y[k] - y_old[k])**2
#         metric = np.sqrt(metric)

#         print(f"[Inner iter {inner_iter+1}] Metric = {metric:.6e}")

#         if metric < inner_tol:
#             print(f"✅ Inner loop converged at iteration {inner_iter+1}.")
#             break
#     else:
#         print(f"⚠️ Inner loop reached max iterations without full convergence.")

#     return A


def update_y_k(delta_k, B, constants, k):
    """
    Closed-form update for auxiliary variable y_k for user k.

    y_k = (sigma_w^2 * I + sum_{n≠k} H_k v_n v_n^H H_k^H )^{-1} H_k v_k
    """

    Nr = constants.NR
    Nt = constants.NT
    K = constants.K
    sigma2 = constants.SIGMA**2   # remember SIGMA is standard deviation, so sigma^2
    Delta_k = delta_k.reshape(Nr, Nt)
    H_k = constants.H_HAT[k] + Delta_k      # (Nr x Nt)
    V = B.V                      # List of (Nr x 1) precoders

    # Build interference covariance matrix
    interference = sigma2 * np.eye(Nr, dtype=complex) 
    for n in range(K):
        if n != k:
            v_n = V[n]              
            interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T.real  # (Nr x Nr)

    # Compute y_k
    v_k = V[k]  # (Nr x 1)
    y_k = np.linalg.solve(interference, H_k @ v_k)  # (Nr x 1)

    return y_k

def update_delta_k(y_k, delta_k, B, constants, k):
    """
    Closed-form update for delta_k given fixed y_k, B, constants for user k.
    c_k &= 1 + 2\Re\{\y_k^H\H_k\v_k\} 
						- \y_k^H(\sigma_w^2\I_{N_R} 
						+ \sum_{n\neq m} \H_k\v_n\v_n^H\H_k^H)\y_k\\
    \a_k &= (\v_n^T\otimes\I_{N_R})^H \y_k \in \mbC^{N_TN_R\times1}
    \nabla_{\bdelta_k} f_k &= (\a_k - \sum_{n\neq k} (\a_n^H\bdelta_k)\a_n)/c_k

    \bdelta_k^\star = - \frac{1}{\lambda_k}\B^{-1} \nabla f_k(\bdelta_k^{(k)}) \in \mbC^{N_TN_R}
    """

    Nr = constants.NR
    Nt = constants.NT
    K = constants.K
    sigma2 = constants.SIGMA**2

    H_hat_k = constants.H_HAT[k]  # (Nr x Nt)
    V = B.V  # List of (Nr x 1) precoders
    B_mat = constants.B  # (Nt*Nr x Nt*Nr)
    lambda_k = B.LAMB[k]  

    # === Step 1: Rebuild H_k ===
    Delta_k = delta_k.reshape(Nr, Nt)  # Reshape (Nt*Nr,1) vector into (Nr,Nt) matrix
    H_k = H_hat_k + Delta_k

    # === Step 2: Compute c_k ===
    vk = V[k]  # (Nr x 1)

    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(K):
        if n != k:
            vn = V[n]
            interference += H_k @ (vn @ vn.conj().T) @ H_k.conj().T

    c_k = 1 \
        + 2 * np.real(y_k.conj().T @ H_k @ vk) \
        - (y_k.conj().T @ interference @ y_k)

    c_k = c_k.item()  # ensure scalar

    # === Step 3: Compute all alpha_n ===
    alpha_list = []
    for n in range(K):
        v_n = V[n]  # (Nr x 1)
        alpha_n = (np.kron(v_n.T, np.eye(Nr)).conj().T) @ y_k  # (Nt*Nr x 1)
        alpha_list.append(alpha_n)

    # === Step 4: Compute gradient ===
    grad = alpha_list[k]
    for n in range(K):
        if n != k:
            grad -= (alpha_list[n].conj().T @ delta_k) * alpha_list[n]
    grad /= c_k

    # === Step 5: Update delta_k ===
    delta_k_star = -np.linalg.solve(B_mat, grad) / lambda_k  # (Nt*Nr x 1)

    return delta_k_star

def compute_rate_k(A, B, constants, k):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2

    H_hat_k = constants.H_HAT[k]
    delta_k = A.delta[k].reshape(Nr, Nt)
    H_k = H_hat_k + delta_k  # effective channel

    V = B.V  # list of v_n

    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            v_n = V[n]  # (Nr x 1)
            interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

    v_k = V[k]  # (Nr x 1)
    SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.inv(interference) @ H_k @ v_k
    SINR_numerator = np.real(SINR_numerator.item())  # ensure scalar real

    rate_k = np.log2(1 + SINR_numerator)

    return rate_k

def compute_lagrangian_A(A, B, constants):
    """
    Compute the total Lagrangian sum_k L1_k for A variables given B and constants.
    """
    K = constants.K
    lagrangian_total = 0

    for k in range(K):
        rate_k = compute_rate_k(A, B, constants, k)
        delta_k = A.delta[k]  # (Nt*Nr, 1)
        B_mat = constants.B
        lambda_k = B.LAMB[k]

        penalty = np.real(delta_k.conj().T @ B_mat @ delta_k) - 1  # scalar
        penalty = penalty.item()

        lagrangian_k = rate_k + lambda_k * penalty

        lagrangian_total += lagrangian_k

    return lagrangian_total
