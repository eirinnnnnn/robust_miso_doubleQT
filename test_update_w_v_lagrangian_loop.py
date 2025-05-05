import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_w, update_v, compute_lagrangian_B

def test_update_w_v_lagrangian_loop():
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16)
    A = VariablesA(constants)
    B = VariablesB(constants)

    print("=== Testing w-v updates until Lagrangian converges ===")

    K = constants.K
    max_iters = 50
    lagrangian_tol = 1e-4

    v_power_trajectory = [[] for _ in range(K)]
    lagrangian_trajectory = []

    for iter in range(max_iters):
        print(f"--- Micro Iteration {iter+1} ---")

        # Save old values
        old_V = [v.copy() for v in B.V]
        old_W = [w.copy() for w in B.W]

        # === Update w ===
        for k in range(K):
            B.W[k] = update_w(A, B, constants, k)

        # === Update v (synchronous)
        new_V = [update_v(A, B, constants, n) for n in range(K)]
        for n in range(K):
            B.V[n] = new_V[n]
            v_power_trajectory[n].append(np.linalg.norm(B.V[n])**2)        

        # === Compute Lagrangian
        L_new = compute_lagrangian_B(A, B, constants)
        lagrangian_trajectory.append(L_new)
        print(f"Lagrangian = {L_new:.6e}")

        if iter > 0 and abs(L_new - lagrangian_trajectory[-2]) < lagrangian_tol:
            print(f"✅ Converged at iteration {iter+1}")
            break
    
        for k in range(K):
            print(f"[Check] ||v_{k}|| = {np.linalg.norm(B.V[k]):.2e}, ||w_{k}|| = {np.linalg.norm(B.W[k]):.2e}")
            # print()

    # === Plot power evolution
    plt.figure()
    for k in range(K):
        plt.plot(range(1, len(v_power_trajectory[k]) + 1), v_power_trajectory[k], marker='o', label=f'$||v_{k}||^2$')
    plt.xlabel("Micro-iteration")
    plt.ylabel("Signal Power")
    plt.title("Power Evolution of Precoder Norms")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("v_power_evolution.png")

    # === Plot Lagrangian
    plt.figure()
    plt.plot(range(1, len(lagrangian_trajectory) + 1), lagrangian_trajectory, marker='x')
    plt.xlabel("Micro-iteration")
    plt.ylabel("Lagrangian Value")
    plt.title("Lagrangian Evolution (B-step)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lagrangian_B_evolution.png")
    plt.show()

if __name__ == "__main__":
    test_update_w_v_lagrangian_loop()
