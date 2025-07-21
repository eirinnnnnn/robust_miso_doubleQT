import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
import copy

# === Core Update Functions ===
def update_A(A, B, constants):
    K = constants.K
    delta_old = [d.copy() for d in A.delta]
    y_old = [y.copy() for y in A.y]

    for k in range(K):
        A.y[k] = update_y_k(A.delta[k], B, constants, k)
        A.delta[k] = update_delta_k(A.y[k], A.delta[k], B, constants, k)

    metric = np.sqrt(sum(
        np.linalg.norm(A.delta[k] - delta_old[k])**2 +
        np.linalg.norm(A.y[k] - y_old[k])**2 for k in range(K)
    ))
    return A, metric

def update_A_loop(A, B, constants, inner_tol=1e-3, max_inner_iter=100, plot_lagrangian=False):
    prev_lagrangian = None
    lag_arr = []
    for inner_iter in range(max_inner_iter):
        for k in range(constants.K):
            A.y[k] = update_y_k(A.delta[k], B, constants, k)
            A.delta[k] = update_delta_k(A.y[k], A.delta[k], B, constants, k)

        current_lagrangian = compute_lagrangian_A(A, B, constants)
        lag_arr.append(current_lagrangian)

        if prev_lagrangian is not None and abs(current_lagrangian - prev_lagrangian) < inner_tol:
            print(f"    ➷ Inner A loop converged at iteration {inner_iter+1}.")
            break
        prev_lagrangian = current_lagrangian
    else:
        print(f"⚠️ Inner A loop reached max iterations without full convergence.")

    if plot_lagrangian:
        plt.plot(range(1, len(lag_arr)+1), lag_arr, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Lagrangian Value')
        plt.title('Lagrangian Trajectory (A loop)')
        plt.grid(True)
        plt.savefig("L1.png")

    return A, True

def update_y_k(delta_k, B, constants, k):
    Nr, Nt, K = constants.NR, constants.NT, constants.K
    sigma2 = constants.SIGMA**2
    Delta_k = delta_k.reshape(Nr, Nt)
    H_k = constants.H_HAT[k] + Delta_k
    V = B.V

    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(K):
        if n != k:
            interference += H_k @ V[n] @ V[n].conj().T @ H_k.conj().T.real

    rhs = H_k @ V[k]
    try:
        y_k = np.linalg.solve(interference, rhs)
    except np.linalg.LinAlgError:
        y_k = np.linalg.pinv(interference) @ rhs

    return y_k

def update_delta_k(y_k, delta_k, B, constants, k):
    Nr, Nt, K = constants.NR, constants.NT, constants.K
    H_k = constants.H_HAT[k] + delta_k.reshape(Nr, Nt)
    h_k = H_k.flatten()
    c_k = compute_ck_QT(delta_k, B, constants, k)

    alpha_list = [
        (np.kron(B.V[n].T, np.eye(Nr)).conj().T) @ y_k for n in range(K)
    ]
    grad = alpha_list[k] - sum((alpha_list[n].conj().T @ h_k) * alpha_list[n] for n in range(K) if n != k)
    grad /= c_k

    delta_k = -np.linalg.solve(constants.B, grad)
    delta_k /= np.sqrt(np.real(grad.conj().T @ constants.Binv @ grad))
    return delta_k

def compute_ck_QT(delta_k, B, constants, k):
    Nr, Nt = constants.NR, constants.NT
    sigma2 = constants.SIGMA ** 2
    delta_k = delta_k.reshape(Nr, Nt)
    H_k = constants.H_HAT[k] + delta_k
    V = B.V

    R_int = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            R_int += H_k @ V[n] @ V[n].conj().T @ H_k.conj().T

    rhs = H_k @ V[k]
    try:
        y_k = np.linalg.solve(R_int, rhs)
    except np.linalg.LinAlgError:
        y_k = np.linalg.pinv(R_int) @ rhs

    signal_term = 2 * np.real(y_k.conj().T @ H_k @ V[k])
    interference_term = np.real(y_k.conj().T @ R_int @ y_k)
    return 1 + signal_term - interference_term

def compute_lagrangian_A(A, B, constants):
    total = 0
    for k in range(constants.K):
        rate_k = compute_rate_k(A, B, constants, k)
        penalty = np.real(A.delta[k].conj().T @ constants.B @ A.delta[k]) - 1
        lagrangian_k = rate_k + B.LAMB[k] * penalty
        total += lagrangian_k
    return total

def compute_rate_k(A, B, constants, k):
    Nr, Nt = constants.NR, constants.NT
    sigma2 = constants.SIGMA**2
    H_k = constants.H_HAT[k] + A.delta[k].reshape(Nr, Nt)

    interference = sigma2 * np.eye(Nr, dtype=complex)
    for n in range(constants.K):
        if n != k:
            interference += H_k @ B.V[n] @ B.V[n].conj().T @ H_k.conj().T

    v_k = B.V[k]
    SINR = np.real(v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k).item()
    return np.log2(1 + SINR)

def generate_delta_within_ellipsoid(Nr, Nt, B):
    dim = Nr * Nt
    z = (np.random.randn(dim) + 1j * np.random.randn(dim))
    z = z / np.linalg.norm(z)
    z = z.reshape(-1, 1)
    quad_norm = np.real(z.conj().T @ B @ z).item()
    r_max = 1 / np.sqrt(quad_norm + 1e-12)
    r = np.random.rand() ** (1 / dim) * r_max
    return (r * z).reshape(Nr, Nt)

def compute_rate_test(A, B, constants, samp=1000):
    rates = []
    Nr, Nt, sigma2 = constants.NR, constants.NT, constants.SIGMA**2
    for _ in range(samp):
        for k in range(constants.K):
            H_k = constants.H_HAT[k] + generate_delta_within_ellipsoid(Nr, Nt, constants.B)
            interference = sigma2 * np.eye(Nr, dtype=complex)
            for n in range(constants.K):
                if n != k:
                    interference += H_k @ B.V[n] @ B.V[n].conj().T @ H_k.conj().T
            v_k = B.V[k]
            SINR = np.real(v_k.conj().T @ H_k.conj().T @ np.linalg.pinv(interference) @ H_k @ v_k).item()
            rates.append(np.log2(1 + SINR))
    return np.mean(rates), np.var(rates)

def compute_g1_k(A, B, constants, k):
    gamma_k = compute_rate_k(A, B, constants, k)
    delta_k = A.delta[k]
    penalty = B.LAMB[k] * (np.vdot(delta_k, constants.B @ delta_k).real - 1)
    return np.log(1 + gamma_k) + penalty
