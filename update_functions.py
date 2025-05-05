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

def update_w(A, B, constants, k):
    """
    Update auxiliary variable w_k given fixed V.
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2

    H_hat_k = constants.H_HAT[k]  # (Nr x Nt)
    delta_k = A.delta[k].reshape(Nr, Nt)
    H_k = H_hat_k + delta_k  
    V = B.V  # List of (Nr x 1) precoders

    # Build interference covariance matrix
    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            v_n = V[n]  # (Nr x 1)
            interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T
    epsilon = 1e-5  
    interference += epsilon * np.eye(interference.shape[0], dtype=complex)

    # Solve for w_k
    v_k = V[k]  # (Nr x 1)
    # w_k = np.linalg.solve(interference, H_k @ v_k)  # (Nr x 1)
    try:
        w_k = np.linalg.solve(interference, H_k @ v_k)
    except np.linalg.LinAlgError:
        w_k = np.linalg.pinv(interference) @ (H_k @ v_k)

    assert_finite(w_k, f"w_{k}")

    return w_k

def update_v(A, B, constants, n):
    """
    Update precoder v_n given fixed w and constants.
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2

    alpha = B.ALPHA
    beta = B.BETA
    K = constants.K

    # Step 1: reconstruct H_n
    H_hat_n = constants.H_HAT[n]
    delta_n = A.delta[n].reshape(Nr, Nt)
    H_n = H_hat_n + delta_n
    W = B.W

    # Step 2: build interference matrix
    interference = beta * np.eye(Nt, dtype=complex)  # Start from beta * I

    for k in range(K):
        if k != n:
            H_hat_k = constants.H_HAT[k]
            delta_k = A.delta[k].reshape(Nr, Nt)
            H_k = H_hat_k + delta_k
            gamma_k = compute_gamma_k(A, B, constants, k)
            QAQ = H_k.conj().T@ W[k] @ W[k].conj().T @ H_k
            # print(H_k.shape, W[k].shape, QAQ.shape)
            interference -= (alpha / gamma_k) * QAQ 
            # interference -= (alpha / gamma_k) * (W[k].conj().T @ H_k @ H_k.conj().T@ W[k])

    # Step 3: compute gamma_n
    gamma_n = compute_gamma_k(A, B, constants, n)

    # Step 4: solve for v_n
    right_hand_side = H_n.conj().T @ W[n]  # (Nt x 1)

    epsilon = 1e-8  
    interference += epsilon * np.eye(interference.shape[0], dtype=complex)

    # v_n = (alpha / gamma_n) * np.linalg.solve(interference, right_hand_side)
    v_n = (alpha / gamma_n) * np.linalg.pinv(interference) @ right_hand_side
    assert_finite(v_n, f"v_{n}")


    return v_n
def compute_gamma_k(A, B, constants, k):
    """
    Compute gamma_k for user k given current A, B, constants.
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    K = constants.K

    H_hat_k = constants.H_HAT[k]
    delta_k = A.delta[k].reshape(Nr, Nt)
    H_k = H_hat_k + delta_k

    W = B.W
    V = B.V

    w_k = W[k]  # (Nr x 1)
    v_k = V[k]  # (Nr x 1)

    # Step 1: Compute 2*Re(w_k^H H_k v_k)
    term1 = 2 * np.real(w_k.conj().T @ H_k @ v_k).item()

    # Step 2: Compute w_k^H (sigma^2 I + interference) w_k
    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(K):
        if n != k:
            H_hat_n = constants.H_HAT[n]
            delta_n = A.delta[n].reshape(Nr, Nt)
            H_n = H_hat_n + delta_n
            v_n = V[n]

            interference += H_n @ (v_n @ v_n.conj().T) @ H_n.conj().T

    term2 = (w_k.conj().T @ interference @ w_k).item()  # scalar

    # Step 3: Combine
    gamma_k = 1 + term1.real - term2.real
    # if gamma_k <= 1e-6:
    #     print(f"⚠️ gamma_k[{k}] too small or negative: {gamma_k}")
    #     gamma_k = 1e-6
    return gamma_k

def update_B_loop(A, B, constants, outer_tol=1e-6, max_outer_iter=10, inner_tol=1e-4, max_inner_iter=500):
    """
    Outer loop for updating B (V, Lambda, t, alpha, beta).
    Inside each outer iteration, run a micro-inner loop on (w, v) until micro-lagrangian converges.
    """

    K = constants.K
    Nr = constants.NR
    Pt = constants.PT

    prev_outer_lagrangian = None
    lagrangian_trajectory = []
    converged = False

    for outer_iter in range(max_outer_iter):
        # === Micro inner loop: update (w, v) alternately ===
        prev_micro_lagrangian = None
        # for micro_iter in range(max_inner_iter):
        #     # Update w
        #     for k in range(K):
        #         w_k = update_w(A, B, constants, k)
        #         B.W[k] = w_k  # Store updated w_k

        #     # Update v
        #     for n in range(K):
        #         B.V[n] = update_v(A, B, constants, n)

        #     # Compute Lagrangian after w,v update
        #     current_micro_lagrangian = compute_lagrangian_B(A, B, constants)
        #     print(f"    [Micro iter {micro_iter+1}] Micro-lagrangian = {current_micro_lagrangian:.6e}")

        #     if prev_micro_lagrangian is not None:
        #         delta_micro = abs(current_micro_lagrangian - prev_micro_lagrangian)
        #         print(f"        Micro Lagrangian change = {delta_micro:.6e}")
        #         if delta_micro < inner_tol:
        #             print(f"    ✅ Micro-inner loop converged at iteration {micro_iter+1}.")
        #             break

        #     prev_micro_lagrangian = current_micro_lagrangian
        B, gamma_history, _ = inner_update_w_v(A, B, constants, max_iter=max_inner_iter, tol=inner_tol)
        gamma_history = np.array(gamma_history)  # shape: (iterations, K)
        for k in range(constants.K):
            print(f"[B inner Final] γ_k[{k}] = {gamma_history[-1, k]:.4e}, w_k norm = {np.linalg.norm(B.W[k]):.4e}, v_k norm = {np.linalg.norm(B.V[k]):.4e}")
        
        # === Step 2: Update lagrangian multipliers ===
        B = update_lagrangian_variables(A, B, constants)

        # === Step 3: Compute outer lagrangian ===
        current_outer_lagrangian = compute_lagrangian_B(A, B, constants)
        lagrangian_trajectory.append(current_outer_lagrangian)

        print(f"[B Outer iter {outer_iter+1}] Outer Lagrangian = {current_outer_lagrangian:.6e}")

        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            print(f"    Outer Lagrangian change = {delta_outer:.6e}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory


def compute_g1_k(A, B, constants, k):
    """
    Compute the full quadratic-transformed utility g_{1,k}^{QT} for user k:
        g_{1,k}^{QT} = log(1 + gamma_k) + lambda_k (delta_k^H B delta_k - 1)

    Args:
        A: VariablesA object (contains delta_k)
        B: VariablesB object (contains W, V, LAMB, and B matrix)
        constants: GlobalConstants object
        k: user index

    Returns:
        g1_k: float, the value of the transformed rate penalty term for user k
    """
    gamma_k = compute_gamma_k(A, B, constants, k)
    delta_k = A.delta[k]  # shape (Nt*Nr, 1)
    lambda_k = B.LAMB[k]
    B_matrix = constants.B

    # First term: log(1 + gamma_k), with domain protection
    if np.real(gamma_k) < -1.0:
        log_term = -np.inf
    else:
        log_term = np.log(np.real(gamma_k))

    # Second term: lambda_k (delta^H B delta - 1)
    penalty_term = lambda_k * (np.vdot(delta_k, B_matrix @ delta_k).real - 1)

    return log_term + penalty_term

def compute_lagrangian_B(A, B, constants):
    """
    Compute the full Lagrangian L2 for B update.
    """

    K = constants.K
    Pt = constants.PT
    sigma2 = constants.SIGMA**2

    t = B.t
    alpha = B.ALPHA
    beta = B.BETA
    lamb = B.LAMB

    total_g1 = 0
    for k in range(K):
        # === Build H_k
        Nr = constants.NR
        Nt = constants.NT
        H_hat_k = constants.H_HAT[k]
        delta_k = A.delta[k].reshape(Nr, Nt)
        H_k = H_hat_k + delta_k

        v_k = B.V[k]
        w_k = B.W[k]

        # # === Compute SINR-related term
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

        # log_arg = 1+SINR_k
        # g1_k = np.log(log_arg)

        # # === Add the penalty term (delta_k constraint)
        # delta_penalty = (delta_k.reshape(-1,1).conj().T @ constants.B @ delta_k.reshape(-1,1)).real.item() - 1

        # g1_k += lamb[k] * delta_penalty

        # total_g1 += g1_k
        total_g1 = 0
        for k in range(K):
            g1_k = compute_g1_k(A, B, constants, k)
            total_g1 += g1_k

    # === Power penalty term
    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))

    # === Full Lagrangian
    lagrangian = t + alpha * (total_g1 - t) + beta * (Pt - total_power)

    return lagrangian
def update_lagrangian_variables(A, B, constants, eta_t=0.01, eta_lambda=0.01, eta_alpha=0.01, eta_beta=0.01):
    """
    Perform gradient ascent on t, lambda_k, alpha, beta.
    """

    K = constants.K
    Pt = constants.PT
    sigma2 = constants.SIGMA**2

    t = B.t
    alpha = B.ALPHA
    beta = B.BETA
    lamb = B.LAMB

    # === Step 1: compute g1_k for each user
    # g1_list = []
    total_g1 = 0
    for k in range(K):
        Nr = constants.NR
        Nt = constants.NT
        H_hat_k = constants.H_HAT[k]
        delta_k = A.delta[k].reshape(Nr, Nt)
        H_k = H_hat_k + delta_k

        v_k = B.V[k]
        w_k = B.W[k]

        for a in range(K):
            g1_k = compute_g1_k(A, B, constants, a)
            total_g1 += g1_k
    #     interference = sigma2 * np.eye(Nr, dtype=complex)
    #     for n in range(K):
    #         if n != k:
    #             H_hat_n = constants.H_HAT[n]
    #             delta_n = A.delta[n].reshape(Nr, Nt)
    #             H_n = H_hat_n + delta_n
    #             v_n = B.V[n]
    #             interference += H_n @ (v_n @ v_n.conj().T) @ H_n.conj().T

    #     SINR_k = np.real(w_k.conj().T @ H_k @ v_k + v_k.conj().T @ H_k.conj().T @ w_k \
    #             - w_k.conj().T @ interference @ w_k).item()

    #     g1_k = np.log(1 + SINR_k)

    #     delta_penalty = (delta_k.reshape(-1,1).conj().T @ constants.B @ delta_k.reshape(-1,1)).real.item() - 1
    #     g1_k += lamb[k] * delta_penalty

    #     g1_list.append(g1_k)

    # total_g1 = sum(g1_list)

    # === Step 2: update t
    B.t = B.t + eta_t * (1 - alpha)

    # === Step 3: update each lambda_k
    for k in range(K):
        delta_penalty = (A.delta[k].reshape(-1,1).conj().T @ constants.B @ A.delta[k].reshape(-1,1)).real.item() - 1
        B.LAMB[k] = max(0, B.LAMB[k] + eta_lambda * alpha * delta_penalty)

    # === Step 4: update alpha
    B.ALPHA = max(0, B.ALPHA - eta_alpha * (total_g1 - B.t))

    # === Step 5: update beta
    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))
    B.BETA = max(0, B.BETA - eta_beta * (Pt - total_power))

    return B
def assert_finite(x, label="variable"):
    if not np.all(np.isfinite(x)):
        print(f"❌ {label} is not finite!")
        print(x)
        raise ValueError(f"{label} became NaN or inf")

def inner_update_w_v(A, B, constants, max_iter=50000, tol=1e-5):
    """
    Alternating updates of (w_k, v_k) for all users until convergence of all gamma_k.

    Args:
        A: VariablesA object
        B: VariablesB object
        constants: GlobalConstants object
        max_iter: maximum number of outer iterations
        tol: stopping threshold for convergence (applied to total change in gamma)

    Returns:
        B: updated VariablesB object (with new W and V)
        gamma_history: list of gamma_k vectors per iteration, shape = (iter, K)
    """
    K = constants.K
    gamma_history = []
    prev_gamma = np.zeros(K)
    signal_log = []

    for i in range(max_iter):
        current_gamma = np.zeros(K)
        current_signal = np.zeros(K)

        w_k_arr = []
        v_k_arr = []
        for k in range(K):
            w_k = update_w(A, B, constants, k)
            w_k_arr.append(w_k)
        B.W = w_k_arr
        for k in range(K):
            v_k = update_v(A, B, constants, k)
            v_k_arr.append(v_k)
        B.V = v_k_arr
        ##signal monitor
        for k in range(K):
            H_hat_k = constants.H_HAT[k]
            delta_k = A.delta[k].reshape(constants.NR, constants.NT)
            H_k = H_hat_k + delta_k
            signal_term = np.real(np.vdot(w_k, H_k @ v_k)).item()
            
            gamma_k = compute_gamma_k(A, B, constants, k)
            current_gamma[k] = gamma_k.real if np.iscomplexobj(gamma_k) else gamma_k
            current_signal[k] = signal_term
        B.V = v_k_arr

        gamma_history.append(current_gamma.copy())
        signal_log.append(current_signal.copy())
        delta = np.linalg.norm(current_gamma - prev_gamma)

        if delta < tol:
            break
        prev_gamma = current_gamma

    return B, gamma_history, signal_log