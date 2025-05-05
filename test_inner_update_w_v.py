import numpy as np
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from update_functions import compute_gamma_k, update_w, update_v
import matplotlib.pyplot as plt



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

def test_inner_update_w_v_loop():
    # === Initialization ===
    constants = GlobalConstants(Nt=16, Nr=2, K=2, Pt=2)    
    A = VariablesA(constants)
    B = VariablesB(constants)
    B = initialize_t(A, B, constants)

    # === Run inner update for all users ===
    B, gamma_history, signal_log = inner_update_w_v(A, B, constants, max_iter=10000, tol=1e-5)
    gamma_history = np.array(gamma_history)  # shape: (iterations, K)
    signal_log= np.array(signal_log)  # shape: (iterations, K)

    print(f"✅ Finished in {gamma_history.shape[0]} iterations")
    
    # === Final gamma values ===
    for k in range(constants.K):
        print(f"[Final] γ_k[{k}] = {gamma_history[-1, k]:.4e}, w_k norm = {np.linalg.norm(B.W[k]):.4e}, v_k norm = {np.linalg.norm(B.V[k]):.4e}")

    # === Plotting gamma convergence ===
    for k in range(constants.K):
        plt.plot(gamma_history[:, k], marker='o', label=f'γ_k[{k}]')

    plt.xlabel("Iteration")
    plt.ylabel("γ_k value")
    plt.title("Convergence of γ_k for all users")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("gamma_convergence.png")

    plt.figure()

    for k in range(constants.K):
        plt.plot(signal_log[:, k], marker='o', label=f'user{k}')

    plt.xlabel("Iteration")
    plt.ylabel("$\Re \{\mathbf{w}_k^H\mathbf{H}_k \mathbf{v}_k\}$")
    plt.title("Signal value for all users")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("signal_term.png")

if __name__ == "__main__":
    test_inner_update_w_v_loop()