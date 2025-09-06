import numpy as np
import sys
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
def update_A_loop(A, B, constants, inner_tol=1e-3, max_inner_iter=100, plot_lagrangian=False):
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
        # print(f"[Inner iter {inner_iter+1}] Lagrangian = {current_lagrangian:.6e}")

        if prev_lagrangian is not None:
            delta_lagrangian = abs(current_lagrangian - prev_lagrangian)
            # print(f"inner Lagrangian change = {delta_lagrangian:.6e}")
            if delta_lagrangian < inner_tol:
                # print(f"    ⤷ Inner A loop converged at iteration {inner_iter+1}.")
                flag = True
                break

        prev_lagrangian = current_lagrangian
    else:
        print(f"⚠️ Inner A loop reached max iterations without full convergence.")
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
    rhs = H_k @ v_k  # (Nr x 1)
    try:
        y_k = np.linalg.solve(interference, rhs)
    except np.linalg.LinAlgError:
        y_k = np.linalg.pinv(interference) @ rhs

    return y_k
def compute_ck(delta_k, B, constants, k):
    """
    Original SINR-based c_k: 
    c_k = 1 + v_k^H H^H (sigma^2 I + interference)^-1 H v_k
    """
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA ** 2

    delta_k = delta_k.reshape(Nr, Nt)
    H_k = constants.H_HAT[k] + delta_k
    V = B.V
    v_k = V[k]

    # Construct interference matrix
    R_int = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            v_n = V[n]
            R_int += H_k @ v_n @ v_n.conj().T @ H_k.conj().T

    # Compute c_k
    Hv = H_k @ v_k
    c_k = 1 + np.real(v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(R_int) @ Hv).item()

    return c_k
def compute_ck_QT(delta_k, B, constants, k):
    """
    QT-based surrogate c_k:
    c_k = 1 + 2 Re{ y^H H v } - y^H (sigma^2 I + interference) y
    """
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA ** 2

    delta_k = delta_k.reshape(Nr, Nt)
    H_k = constants.H_HAT[k] + delta_k
    V = B.V
    v_k = V[k]

    # Construct interference matrix
    R_int = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            v_n = V[n]
            R_int += H_k @ v_n @ v_n.conj().T @ H_k.conj().T

    # Compute y_k = R_int^{-1} H_k v_k
    rhs = H_k @ v_k
    try:
        y_k = np.linalg.solve(R_int, rhs)
    except np.linalg.LinAlgError:
        y_k = np.linalg.pinv(R_int) @ rhs

    # Compute c_k_QT
    signal_term = 2 * np.real(y_k.conj().T @ H_k @ v_k)
    interference_term = np.real(y_k.conj().T @ R_int @ y_k)
    c_k_QT = 1 + signal_term - interference_term

    return c_k_QT.item()


def update_delta_k(y_k, delta_k, B, constants, k):
    """
    Closed-form update for delta_k given fixed y_k, B, constants for user k.
    c_k &= 1 + 2\Re\{\y_k^H\H_k\v_k\} 
						- \y_k^H(\sigma_w^2\I_{N_R} 
						+ \sum_{n\neq m} \H_k\v_n\v_n^H\H_k^H)\y_k\\
    \a_k &= (\v_n^T\otimes\I_{N_R})^H \y_k \in \mbC^{N_TN_R\times1}
    \nabla_{\bdelta_k} f_k &= (\a_k - \sum_{n\neq k} (\a_n^H\bdelta_k)\a_n)/c_k

    \bdelta_k^\star = - \B^{-1} \nabla f_k(\bdelta_k^{(k)}) \in \mbC^{N_TN_R}
    """

    Nr = constants.NR
    Nt = constants.NT
    K = constants.K
    sigma2 = constants.SIGMA**2

    H_hat_k = constants.H_HAT[k]  # (Nr x Nt)
    V = B.V  # List of (Nr x 1) precoders
    B_mat = constants.B[k]  # (Nt*Nr x Nt*Nr)
    # === Step 1: Rebuild H_k ===
    Delta_k = delta_k.reshape(Nr, Nt)  # Reshape (Nt*Nr,1) vector into (Nr,Nt) matrix
    H_k = H_hat_k + Delta_k
    h_k = H_k.flatten()

    c2 = compute_ck_QT(delta_k, B, constants, k)
    c_k = c2

    alpha_list = []
    for n in range(K):
        v_n = V[n]  # (Nr x 1)
        alpha_n = (np.kron(v_n.T, np.eye(Nr)).conj().T) @ y_k  # (Nt*Nr x 1)
        alpha_list.append(alpha_n)

    # === Step 4: Compute gradient ===
    grad = alpha_list[k]
    for n in range(K):
        if n != k:
            grad -= (alpha_list[n].conj().T @ h_k) * alpha_list[n]
    grad /= c_k

    # === Step 5: Update delta_k ===
    quad_form = np.real(delta_k.conj().T @ constants.B[k] @ delta_k).item()
    norm = np.real(grad.conj().T @ constants.Binv[k] @ grad).item()

    delta_k = -np.linalg.solve(B_mat, grad)   # (Nt*Nr x 1)

    delta_k /= np.sqrt(norm)

    # print(f"delta_k[{k}] norm: {np.linalg.norm(delta_k):.4e}, grad_norm = {np.linalg.norm(grad):.4e}, quad_norm = {quad_form:.4e}")
    return delta_k
import numpy as np

def generate_mmse_mismatch(Nr, Nt, B):
    """
    Generate a complex matrix delta of shape (Nr, Nt) such that
    vec(delta).H @ B @ vec(delta) <= 1
    """
    dim = Nr * Nt

    # Step 1: sample from standard complex normal
    z_real = np.random.randn(dim)
    z_imag = np.random.randn(dim)
    z = z_real + 1j * z_imag
    z = z / np.linalg.norm(z)  # normalize to unit norm

    # Step 2: sample radius uniformly inside unit ball (w.r.t. B)
    # Define scaling factor r such that (r*z)^H B (r*z) = r^2 * z^H B z <= 1
    # Thus: r <= 1 / sqrt(z^H B z)
    z = z.reshape(-1, 1)
    quad_norm = np.real(z.conj().T @ B @ z).item()
    r_max = 1 / np.sqrt(quad_norm + 1e-12)  # +1e-12 for safety
    r = np.random.rand() ** (1 / dim) * r_max

    delta_vec = r * z  # (Nt*Nr, 1)
    delta = delta_vec.reshape(Nr, Nt)

    return delta
def generate_delta_within_ellipsoid(Nr, Nt, B):
    """
    Generate a complex matrix delta of shape (Nr, Nt) such that
    vec(delta).H @ B @ vec(delta) <= 1
    """
    dim = Nr * Nt

    # Step 1: sample from standard complex normal
    z_real = np.random.randn(dim)
    z_imag = np.random.randn(dim)
    z = z_real + 1j * z_imag
    z = z / np.linalg.norm(z)  # normalize to unit norm

    # Step 2: sample radius uniformly inside unit ball (w.r.t. B)
    # Define scaling factor r such that (r*z)^H B (r*z) = r^2 * z^H B z <= 1
    # Thus: r <= 1 / sqrt(z^H B z)
    z = z.reshape(-1, 1)
    quad_norm = np.real(z.conj().T @ B @ z).item()
    r_max = 1 / np.sqrt(quad_norm + 1e-12)  # +1e-12 for safety
    r = np.random.rand() ** (1 / dim) * r_max

    delta_vec = r * z  # (Nt*Nr, 1)
    delta = delta_vec.reshape(Nr, Nt)

    return delta

from scipy.stats import chi2
def compute_rate_over(A, B, constants):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate = 0
    samp = 1000
    for _ in range(samp):
        for k in range(constants.K):
            H_hat_k = constants.H_HAT[k]
            r2 = chi2.ppf(0.9, df=Nt*Nr*2)
            snrest_db = -10 
            SIGMAEST = (10 ** (-snrest_db / 20))
            eps = (1 + 1 / SIGMAEST**2) / r2 
            BB = eps * np.eye(Nt*Nr)
            delta_k = generate_delta_within_ellipsoid(Nr, Nt, BB) 
            H_k = H_hat_k + delta_k  # effective channel

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate += np.log2(1 + SINR_numerator)
    return rate/samp

def compute_rate_perfect(A, B, constants, samp=1000):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate = 0
    # samp = 1000
    for _ in range(1):
        for k in range(constants.K):
            H_hat_k = constants.H_HAT[k]
            # delta_k = generate_delta_within_ellipsoid(Nr, Nt, constants.B[k]) 
            delta_k = A.Delta[k]
            H_k = H_hat_k 

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate = np.log2(1 + SINR_numerator)
    return rate
def compute_rate_worst(A, B, constants, samp=1000):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate = 0
    # samp = 1000
    for _ in range(1):
        for k in range(constants.K):
            H_hat_k = constants.H_HAT[k]
            # delta_k = generate_delta_within_ellipsoid(Nr, Nt, constants.B[k]) 
            delta_k = A.Delta[k]
            H_k = H_hat_k + delta_k  # effective channel

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate = np.log2(1 + SINR_numerator)
    return rate

def compute_rate_test_random(A, B, constants, samp=1000):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    """

    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate =[] 
    # samp = 1000
    for _ in range(samp):
        for k in range(constants.K):
            H_hat_k = constants.H_HAT[k]
            delta_k = generate_delta_within_ellipsoid(Nr, Nt, constants.B[k]) 
            H_k = H_hat_k + delta_k  # effective channel

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate.append(np.log2(1 + SINR_numerator))
    return np.mean(rate), np.var(rate)

from scipy.io import loadmat
def compute_rate_test(A, B, constants, Delta_k, samp=1000, outage_percentile=5):
    """
    Compute the rate R_k for user k given current A (delta) and B (precoders).
    Load the mismatch from generated samples
    """
    # scale = 10**(constants.SNREST_DB/20)
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate = [] 
    mean = []
    var = []
    outage = []
    # samp = 1000
    # delta_k_data = loadmat('Delta_k.mat')
    # Delta = delta_k_data['Delta_k']
    for k in range(constants.K):
        rate_k = []
        for samp in range(samp):
            scale = np.sqrt(1/ (1 + 10**(constants.SNREST_DB[k]/10)))
            # scale = 1
            H_hat_k = constants.H_HAT[k]
            # delta_k = generate_delta_within_ellipsoid(Nr, Nt, constants.B[k]) 
            delta_k = scale*Delta_k[samp][k]
            H_k = H_hat_k + delta_k  # effective channel

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate_k.append(np.log2(1 + SINR_numerator))
        rate.append(rate_k)
        mean.append(np.mean(rate_k))
        var.append(np.var(rate_k))
        outage.append(np.percentile(rate_k, outage_percentile))
    return rate, mean, var, outage 

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
    SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
    SINR_numerator = np.real(SINR_numerator.item())  

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
        B_mat = constants.B[k]
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

    v_k = V[k]  # (Nr x 1)
    try:
        w_k = np.linalg.solve(interference, H_k @ v_k)
    except np.linalg.LinAlgError:
        w_k = np.linalg.pinv(interference) @ (H_k @ v_k)
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
            gamma_k = compute_gamma_k_QT(A, B, constants, k)
            QAQ = H_k.conj().T@ W[k] @ W[k].conj().T @ H_k
            interference += (alpha / gamma_k) * QAQ 
            # interference -= (alpha / gamma_k) * (W[k].conj().T @ H_k @ H_k.conj().T@ W[k])
    gamma_n = compute_gamma_k_QT(A, B, constants, n)
    right_hand_side = H_n.conj().T @ W[n]  # (Nt x 1)
    inv_interf = np.linalg.pinv(interference) 
    v_n = (alpha / gamma_n) * inv_interf @ right_hand_side

    # if (np.linalg.norm(v_n) ==0 ):
    #     print(f"⚠️ v_n[{n}] vanished: {np.linalg.norm(v_n):.4e}")
    #     # sys.exit(1)
    #     B.V = []

    #     total_power = 0

    #     for _ in range(constants.K):
    #         V_k = np.random.normal(0, 1/np.sqrt(2), (constants.NT, 1)) \
    #               + 1j * np.random.normal(0, 1/np.sqrt(2), (constants.NT, 1))
    #         B.V.append(V_k)
    #         total_power += np.linalg.norm(V_k**2) 
    #         # print(V_k, total_power)

    #     scaling_factor = np.sqrt(constants.PT / total_power)
    #     B.V = [scaling_factor * V_k for V_k in B.V]
    norm = np.linalg.norm(v_n)
    if norm < 1e-10 or not np.all(np.isfinite(v_n)):
        print(f"norm of v_{n}: {norm}, gamma_{n}: {gamma_n:.6e}, H^H_w: {np.linalg.norm(right_hand_side)}, I+N: {np.trace(inv_interf)}")
        raise ValueError(f"v_n[{n}] vanished or became invalid.")
    return v_n

def compute_gamma_k_QT(A, B, constants, k):
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
    H_hat_n = constants.H_HAT[k]
    delta_n = A.delta[k].reshape(Nr, Nt)
    H_n = H_hat_n + delta_n
    for n in range(K):
        if n != k:
            v_n = V[n]
            interference += H_n @ (v_n @ v_n.conj().T) @ H_n.conj().T

    term2 = (w_k.conj().T @ interference @ w_k).item()  # scalar

    gamma_k = 1 + term1.real - term2.real

    return gamma_k

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


    # Step 2: Compute (sigma^2 I + interference)
    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(K):
        if n != k:
            # H_hat_n = constants.H_HAT[n]
            # delta_n = A.delta[n].reshape(Nr, Nt)
            # H_n = H_hat_n + delta_n
            v_n = V[n]

            interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

    # term2 = (w_k.conj().T @ interference @ w_k).item()  # scalar
    interference = np.linalg.pinv(interference)

    # Step 3: Combine
    gamma_k = 1 + np.real(v_k.conj().T @ H_k.conj().T @ interference @ H_k @ v_k).item()
    # if gamma_k <= 1e-6:
    #     print(f"⚠️ gamma_k[{k}] too small or negative: {gamma_k}")
    #     gamma_k = 1e-6

    # if gamma_k < 0:
    #     print(f"[Warning] γ_k[{k}] became negative: γ = {gamma_k:.4e}")
    #     print(f"    term1 = {term1:.4e}, term2 = {term2:.4e}")
    #     print(f"    |w_k|^2 = {np.linalg.norm(w_k):.4e}, |v_k|^2 = {np.linalg.norm(v_k):.4e}")

    return gamma_k
#######################

import numpy as np
import copy

# ... other functions remain unchanged ...

def backtracking_line_search_max(obj_fn, grad_fn, x, d, initial_eta=1.0, alpha=1e-4, beta=0.5):
    """
    Backtracking line search for maximization:
    Finds η such that: f(x + η d) ≥ f(x) + α η ∇f(x)^T d
    """
    eta = initial_eta
    fx = obj_fn(x)
    grad_fx = grad_fn(x)
    directional_derivative = np.real(np.vdot(grad_fx, d))

    while True:
        new_x = x + eta * d
        if obj_fn(new_x) >= fx + alpha * eta * directional_derivative:
            break
        eta *= beta
        if eta < 1e-10:
            break
    return eta

def update_lagrangian_variables_maxmin(A, B, constants, ite, robust=True):
    eta_0 = 1e-1 #(snr = 0, snrest>5)
    eta_beta = eta_0
    # if robust==False:
    #     eta_0 = 1e-4
    #     eta_alpha = eta_0*2
    #     eta_beta = eta_0 * 1e-4

    K = constants.K
    Pt = constants.PT

    sigma2 = constants.SIGMA**2
    beta = B.BETA

    def lagrangian(B_local):
        return compute_lagrangian_B_maxmin(A, B_local, constants)

    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))

    d_beta = np.array([(Pt - total_power)])
    res_beta = Pt - total_power
    eta_beta = eta_beta * abs(res_beta) + 1e-8

    B.BETA = max(0, beta - eta_beta * d_beta[0])

    return B

def update_lagrangian_variables(A, B, constants, ite, robust=True):
    eta_0 = 1e-2 #(snr = 0, snrest>5)
    eta_alpha = eta_0*2
    eta_beta = eta_0 * 2e-2
    # eta_0 = 2e-2 #(snr=0, snrest=3)
    # eta_0 = 1.5e-3 #(snr=0, snrest=0)
    exp_para=1e-2
    if robust==False:
        eta_0 = 1e-4
        eta_alpha = eta_0*2
        eta_beta = eta_0  

    # eta = max(eta_0 * 2**(-exp_para*ite), 1e-7)
    # eta = eta_0/np.sqrt(1+ite) 
    # eta = eta_0
    """
    Perform gradient ascent on t, lambda_k, alpha, beta using backtracking line search.
    """
    K = constants.K
    Pt = constants.PT

    sigma2 = constants.SIGMA**2
    t = B.t
    alpha = B.ALPHA
    beta = B.BETA

    def lagrangian(B_local):
        return compute_lagrangian_B(A, B_local, constants)

    # === Step 1: update t ===
    def obj_t(t_val):
        B_tmp = copy.deepcopy(B)
        B_tmp.t = t_val
        return lagrangian(B_tmp)

    def grad_t(t_val):
        return np.array([1 - alpha])

    d_t = grad_t(t)
    # eta_t = backtracking_line_search_max(obj_t, grad_t, t, d_t)
    # eta_t = 1e-3 
    # eta_t = eta 
    res_t = 1 - alpha
    eta_t = eta_0 * abs(res_t) + 1e-8
    # eta_t = eta_0*2**(-exp_para*ite) + 1e-8
    # eta_t = eta_0 + 1e-8

    B.t += eta_t * d_t[0]

    # === Step 2: update each lambda_k ===
    for k in range(K):
        B.LAMB[k] = 0 
        # B.LAMB[k] = max(0, lamb[k] + eta_lamb * d_lamb[0])

    # === Step 3: update alpha ===
    total_g1 = sum(compute_g1_k(A, B, constants, k) for k in range(K))

    def obj_alpha(alpha_val):
        B_tmp = copy.deepcopy(B)
        B_tmp.ALPHA = alpha_val
        return lagrangian(B_tmp)

    def grad_alpha(alpha_val):
        return np.array([(total_g1 - B.t)])

    d_alpha = grad_alpha(alpha)
    # eta_alpha = backtracking_line_search_max(obj_alpha, grad_alpha, alpha, d_alpha)
    # eta_alpha = eta
    res_alpha = total_g1 - B.t
    eta_alpha = eta_alpha * abs(res_alpha) + 1e-8
    # eta_alpha = eta_0*2**(-exp_para*ite)+ 1e-8


    B.ALPHA = max(0, alpha - eta_alpha * d_alpha[0])
    # B.ALPHA = max(1e-3, min(10, B.ALPHA - eta_alpha * d_alpha[0]))

    # === Step 4: update beta ===
    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))

    def obj_beta(beta_val):
        B_tmp = copy.deepcopy(B)
        B_tmp.BETA = beta_val
        return lagrangian(B_tmp)

    def grad_beta(beta_val):
        return np.array([(Pt - total_power)])

    d_beta = grad_beta(beta)
    # eta_beta = backtracking_line_search_max(obj_beta, grad_beta, beta, d_beta)
    # eta_beta = eta
    res_beta = Pt - total_power
    # eta_beta = 5e-5 * abs(res_beta) + 1e-8
    # eta_beta = 5e-6*2**(-exp_para*ite)  + 1e-8
    eta_beta = eta_beta * abs(res_beta) + 1e-8

    # print(f"grad_beta = {d_beta}, eta_beta = {eta_beta}, beta = {beta}")
    # B.BETA = max(1e-3, min(10, B.BETA - eta_beta * d_beta[0]))
    B.BETA = max(0, beta - eta_beta * d_beta[0])
    # print(f"[Dual Update] alpha = {alpha:.4e}, slack = {total_g1 - B.t:.4e}, η = {eta_alpha:.2e}")
    # print(f"[Dual Update] beta = {beta:.4e}, slack = {Pt-total_power:.4e}, η = {eta_beta:.2e}")
    # print(f"[Prim Update] t = {t:.4e}, η = {eta_t:.2e}")

    return B

def update_B_loop_robust_maxmin(A, B, constants, 
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
    for outer_iter in range(max_outer_iter):

        if robust:
            A, converged_A = update_A_loop(A, B, constants, inner_tol=inner_tol, max_inner_iter=max_inner_iter)

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

        B = update_lagrangian_variables_maxmin(A, B, constants, outer_iter, robust)
        beta_trajectory.append(B.BETA)    

        slack_pwr =  constants.PT - sum(np.linalg.norm(B.V[k]) ** 2 for k in range(constants.K)) 

        res2.append(slack_pwr)

        current_outer_lagrangian = compute_lagrangian_B_maxmin(A, B, constants)
        lagrangian_trajectory.append(current_outer_lagrangian)

        # === Check convergence ===
        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}")
            # print(f"slk_pwr = {slack_pwr:.6e}, beta = {B.BETA:.6e}")

            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}, alpha = {B.ALPHA}, beta = {B.BETA}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2

ddelta_k = loadmat("Delta_k.mat")
ddelta_k = ddelta_k["Delta_k"] 
def update_B_loop_robust_stableB(A, B, constants, 
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
    # if robust==False:
    #     A.Delta = []
    #     A.delta = []
    #     for _ in range(constants.K):
    #         Delta_k = generate_delta_within_ellipsoid(constants.NR, constants.NT, constants.B[k]) 
    #         A.Delta.append(Delta_k)
    #         A.delta.append(Delta_k.reshape(-1, 1))  # flatten into column

    for outer_iter in range(max_outer_iter):

        if robust:
            A, converged_A = update_A_loop(A, B, constants, inner_tol=inner_tol, max_inner_iter=max_inner_iter)

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

        B = update_lagrangian_variables(A, B, constants, outer_iter, robust)
        # lag = compute_lagrangian_B(A, B, constants)
        # lagrangian_trajectory.append(lag)

        alpha_trajectory.append(B.ALPHA)  
        beta_trajectory.append(B.BETA)    
        t_trajectory.append(B.t)

        g_sum = 0
        for k in range(constants.K):
            g_k = compute_g1_k_QT(A, B, constants, k)
            g_sum += g_k

        slack_g = g_sum - B.t
        slack_pwr =  constants.PT - sum(np.linalg.norm(B.V[k]) ** 2 for k in range(constants.K)) 

        res1.append(slack_g)
        res2.append(slack_pwr)

        # === Step 3: Compute outer Lagrangian ===
        current_outer_lagrangian = compute_lagrangian_B(A, B, constants)
        # current_outer_lagrangian = compute_rate_test(A, B, constants, samp=1000)[0]
        lagrangian_trajectory.append(current_outer_lagrangian)

        # === Check convergence ===
        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            # if (robust == False):
                # n, n_m, n_v, n_o = compute_rate_test(A, B, constants, ddelta_k, samp=1000)
                # print(f"rate = {n_m:.6e}, slk_pwr = {slack_pwr:.6e}")
            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}")

            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}, alpha = {B.ALPHA}, beta = {B.BETA}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                # _, n_m, n_v, n_o = compute_rate_test(A, B, constants, ddelta_k, samp=1000)
                # print(f"final rate = {n_m:.6e}, slk_pwr = {slack_pwr:.6e}")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2

def update_B_loop_robust(A, B, constants, 
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
    # if robust==False:
    #     A.Delta = []
    #     A.delta = []
    #     for _ in range(constants.K):
    #         Delta_k = np.zeros((constants.NR, constants.NT)) 
    #         # Delta_k = generate_delta_within_ellipsoid(constants.NR, constants.NT, constants.B[k]) 
    #         A.Delta.append(Delta_k)
    #         A.delta.append(Delta_k.reshape(-1, 1))  
    for outer_iter in range(max_outer_iter):
        # === Step 1: inner w,v update ===
        B, gamma_history, _, sinr_history = inner_update_w_v(A, B, constants, max_iter=max_inner_iter, tol=inner_tol)

        # === Step 2: Lagrangian dual variable update ===
        B = update_lagrangian_variables(A, B, constants, outer_iter)

        if robust:
            A, _= update_A_loop(A, B, constants, inner_tol=inner_tol, max_inner_iter=max_inner_iter)

        # === NEW: Log alpha and beta ===
        alpha_trajectory.append(B.ALPHA)  
        beta_trajectory.append(B.BETA)    
        t_trajectory.append(B.t)


        # Compute g_sum = sum_k g_{1,k}(V)
        g_sum = 0
        for k in range(constants.K):
            g_k = compute_g1_k_QT(A, B, constants, k)
            g_sum += g_k

        slack_g = g_sum - B.t
        slack_pwr =  constants.PT - sum(np.linalg.norm(B.V[k]) ** 2 for k in range(constants.K)) 

        res1.append(slack_g)
        res2.append(slack_pwr)

        # === Step 3: Compute outer Lagrangian ===
        current_outer_lagrangian = compute_lagrangian_B(A, B, constants)
        lagrangian_trajectory.append(current_outer_lagrangian)

        # === Check convergence ===
        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}")

            # print(f"[{outer_iter}]Outer Lagrangian change = {delta_outer:.6e}, alpha = {B.ALPHA}, beta = {B.BETA}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")


    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2
import copy

def modified_update_B_loop_robust(A, B, constants, 
                         outer_tol=1e-6, max_outer_iter=100, 
                         inner_tol=1e-4, max_inner_iter=500,
                         robust=True, Delta_k_id=-1):
    """
    Modified outer loop for updating B with convergence monitoring based on:
    - Outer Lagrangian change
    - Norm change in v_k (Δv)
    - Norm change in delta_k (Δδ)
    Also logs trajectories of alpha, beta, t, Δv, Δδ, and v_k itself.
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
    delta_v_trajectory = []
    delta_delta_trajectory = []
    v_trajectory = []

    converged = False

    # if not robust:
    #     # generate random mismatch for hat_h
    #     # A.Delta = []
    #     # A.delta = []
    #     scale = np.sqrt(1/ (1 + 10**(constants.SNREST_DB/10)))
    #     A.Delta = loadmat("Delta_k.mat")["Delta_k"][Delta_k_id] * scale
    #     # A.Delta = scale*np.array([np.random.normal(0, 1/np.sqrt(2), (Nr, constants.Nt)) + 1j * np.random.normal(0, 1/np.sqrt(2), (Nr, constants.Nt)) for _ in range(K)])
    #     A.delta = [Delta_k.reshape(-1, 1) for Delta_k in A.Delta]
    #     # for _ in range(constants.K):
    #         # Delta_k = generate_delta_within_ellipsoid(Nr, constants.NT, constants.B[k]) 
    #         # A.Delta.append(Delta_k)
    #         # A.delta.append(Delta_k.reshape(-1, 1))  # flatten into column

    for outer_iter in range(max_outer_iter):
        # === Step 1: inner w,v update ===
        prev_V = [v.copy() for v in B.V]
        B, gamma_history, _, sinr_history = inner_update_w_v(A, B, constants, max_iter=max_inner_iter, tol=inner_tol)
        delta_v = sum(np.linalg.norm(B.V[k] - prev_V[k]) for k in range(K))
        delta_v_trajectory.append(delta_v)
        v_trajectory.append(sum(np.linalg.norm(B.V[k]) for k in range(K)))

        # === Step 2: Lagrangian dual variable update ===
        B = update_lagrangian_variables(A, B, constants, outer_iter)

        # === Step 3: A-step ===
        if robust:
            prev_delta = [d.copy() for d in A.delta]
            A, _ = update_A_loop(A, B, constants, inner_tol=inner_tol, max_inner_iter=max_inner_iter)
            delta_delta = sum(np.linalg.norm(A.delta[k] - prev_delta[k]) for k in range(K))
        else:
            delta_delta = 0.0
        delta_delta_trajectory.append(delta_delta)

        # === Log alpha, beta, t ===
        alpha_trajectory.append(B.ALPHA)  
        beta_trajectory.append(B.BETA)    
        t_trajectory.append(B.t)

        # === Constraint slacks ===
        g_sum = sum(compute_g1_k_QT(A, B, constants, k) for k in range(K))
        slack_g = g_sum - B.t
        slack_pwr = Pt - sum(np.linalg.norm(B.V[k]) ** 2 for k in range(K))
        res1.append(slack_g)
        res2.append(slack_pwr)

        # === Compute outer Lagrangian ===
        current_outer_lagrangian = compute_lagrangian_B(A, B, constants)
        lagrangian_trajectory.append(current_outer_lagrangian)

        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            print(f"[{outer_iter}] ΔL = {delta_outer:.2e}, Δv = {delta_v:.2e}, Δδ = {delta_delta:.2e}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory, alpha_trajectory, beta_trajectory, t_trajectory, res1, res2, delta_v_trajectory, delta_delta_trajectory, v_trajectory

########################

def compute_g1_k(A, B, constants, k):
    """
    Compute the full quadratic-transformed utility g_{1,k}^{QT} for user k:
        g_{1,k} = log(1+SINR) + lambda_k (delta_k^H B delta_k - 1)

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
    # lambda_k = B.LAMB[k]
    lambda_k = 0 
    B_matrix = constants.B[k]

    # First term: log(1 + gamma_k), with domain protection
    if np.real(gamma_k) < -1.0:
        log_term = -np.inf
    else:
        log_term = np.log(np.real(gamma_k))

    # Second term: lambda_k (delta^H B delta - 1)
    penalty_term = lambda_k * (np.vdot(delta_k, B_matrix @ delta_k).real - 1)

    return log_term + penalty_term
def compute_g1_k_QT(A, B, constants, k):
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
    gamma_k = compute_gamma_k_QT(A, B, constants, k)
    delta_k = A.delta[k]  # shape (Nt*Nr, 1)
    lambda_k = B.LAMB[k]
    B_matrix = constants.B[k]

    # First term: log(1 + gamma_k), with domain protection
    if np.real(gamma_k) < -1.0:
        log_term = -np.inf
    else:
        log_term = np.log(np.real(gamma_k))

    # Second term: lambda_k (delta^H B delta - 1)
    penalty_term = lambda_k * (np.vdot(delta_k, B_matrix @ delta_k).real - 1)

    return log_term - penalty_term

def compute_lagrangian_B_maxmin(A, B, constants):
    """
    Compute the full Lagrangian L2 for B update.
    """

    K = constants.K
    Pt = constants.PT
    sigma2 = constants.SIGMA**2

    beta = B.BETA
    lamb = B.LAMB

    total_g1 = 0
    for k in range(K):
        total_g1 = 0
        for k in range(K):
            g1_k = compute_g1_k(A, B, constants, k)
            total_g1 += g1_k

    # === Power penalty term
    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))

    # === Full Lagrangian
    lagrangian =  total_g1  + beta * (Pt - total_power)

    return lagrangian
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
        total_g1 = 0
        for k in range(K):
            g1_k = compute_g1_k(A, B, constants, k)
            total_g1 += g1_k

    # === Power penalty term
    total_power = sum(np.linalg.norm(B.V[k])**2 for k in range(K))

    # === Full Lagrangian
    lagrangian = t + alpha * (total_g1 - t) + beta * (Pt - total_power)

    return lagrangian

def assert_finite(x, label="variable"):
    if not np.all(np.isfinite(x)):
        print(f"❌ {label} is not finite!")
        print(x)
        raise ValueError(f"{label} became NaN or inf")

def inner_update_w_v(A, B, constants, max_iter=1000000, tol=1e-5):
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
    sinr_history = []
    prev_gamma = np.zeros(K)
    signal_log = []

    for i in range(max_iter):
        current_gamma = np.zeros(K)
        current_sinr= np.zeros(K)
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

        ## lagrangian parameters update

        ##signal monitor
        for k in range(K):
            H_hat_k = constants.H_HAT[k]
            delta_k = A.delta[k].reshape(constants.NR, constants.NT)
            H_k = H_hat_k + delta_k
            signal_term = np.real(np.vdot(w_k, H_k @ v_k)).item()
            
            gamma_k = compute_gamma_k_QT(A, B, constants, k)
            sinr = compute_gamma_k(A, B, constants, k)

            current_gamma[k] = gamma_k.real if np.iscomplexobj(gamma_k) else gamma_k
            current_sinr[k] = sinr.real if np.iscomplexobj(sinr) else sinr
            current_signal[k] = signal_term
        B.V = v_k_arr

        gamma_history.append(current_gamma.copy())
        sinr_history.append(current_sinr.copy())
        signal_log.append(current_signal.copy())
        delta = np.linalg.norm(current_gamma - prev_gamma)
        pos_gam = np.all(current_gamma > 0)

        if delta < tol and pos_gam:
            break
        prev_gamma = current_gamma
    
    for k in range(K):
        if current_gamma[k] < 0:
            print(f"⚠️ gamma_k[{k}] became negative: {current_gamma[k]:.4e}")
            print(f"    |w_k|^2 = {np.linalg.norm(B.W[k]):.4e}, |v_k|^2 = {np.linalg.norm(B.V[k]):.4e}")
    
    return B, gamma_history, signal_log, sinr_history


def compute_outage_rate(A, B, constants, Delta_k,outage_percentile=5, samp=10000):
    """
    Compute 5%-outage rate over all samples and all users.

    Args:
        H_list: list of true channels for each user in each Monte Carlo run (shape: [Nsamp][K, Nr, Nt])
        V_list: list of precoders (shape: [Nsamp][Nt, K])
        delta_list: list of mismatch matrices (same shape as H_list)
        noise_power: scalar noise variance (default: 1)
        outage_percentile: percentage for outage computation (default: 5)

    Returns:
        outage_rate: scalar value of the 5%-percentile sum rate across samples
    """
    scale = np.sqrt(1/ (1 + 10**(constants.SNREST_DB/10)))
    Nr = constants.NR
    Nt = constants.NT
    sigma2 = constants.SIGMA**2
    rate = 0
    # samp = 1000
    all_sum_rate = []
    sum_rate = 0

    for _ in range(samp):
        for k in range(constants.K):
            H_hat_k = constants.H_HAT[k]
            delta_k = scale*Delta_k[samp][k]
            H_k = H_hat_k + delta_k  # effective channel

            V = B.V  # list of v_n

            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    v_n = V[n]  # (Nr x 1)
                    interference += H_k @ (v_n @ v_n.conj().T) @ H_k.conj().T

            v_k = V[k]  # (Nr x 1)
            SINR_numerator = v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k
            SINR_numerator = np.real(SINR_numerator.item())  

            rate += np.log2(1 + SINR_numerator)
            sum_rate += rate

        all_sum_rate.append(sum_rate)

    # Compute outage percentile
    outage_rate = np.percentile(all_sum_rate, outage_percentile)
    return outage_rate

def wmmse_sum_rate(constants, max_iter=100, tol=1e-4):
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
    H_HAT=constants.H_HAT
    Pt = constants.PT
    sigma2 = constants.SIGMA**2
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