from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_w, update_v
from update_functions import update_A_loop, update_B_loop
import matplotlib.pyplot as plt
import numpy as np
def test_update_w_v_loop():
    # === Step 1: Initialize
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=16, Nr=4, K=4, Pt=100)
    A = VariablesA(constants)
    B = VariablesB(constants)
    # print(B.V[0].shape)
    # print(B.W[0].shape)
    # print(A.delta[0].shape)

    print("=== Testing alternating update_w and update_v ===")

    A, converged_A= update_A_loop(A, B, constants)
    max_micro_iter = 1000 
    inner_tol = 1e-10
    # lagrangian_trajectory = []
    v_power_evolution = []
    for micro_iter in range(max_micro_iter):
        print(f"--- Micro Iteration {micro_iter+1} ---")

        # Backup old w and v to monitor convergence
        w_old = [wk.copy() for wk in B.W]
        v_old = [vk.copy() for vk in B.V]
        w_new= []
        v_new = []
        pwr = 0
        # === Step 2: Update all w_k
        for k in range(constants.K):
            w_k = update_w(A, B, constants, k)
            w_new.append(w_k)    
            w_norm = np.linalg.norm(w_k)
            print(f"[w_{k}] norm = {w_norm:.6e}")
            assert np.isfinite(w_norm), f"w_{k} has non-finite norm!"
        B.W = w_new
        # === Step 3: Update all v_n
        for n in range(constants.K):
            v_n = update_v(A, B, constants, n)
            v_new.append(v_n) 
            v_norm = np.linalg.norm(v_n)
            pwr += v_norm
            print(f"[v_{n}] norm = {v_norm:.6e}")
            assert np.isfinite(v_norm), f"v_{n} has non-finite norm!"
        B.V= v_new
        v_power_evolution.append(pwr)

        # === Step 4: Check convergence (optional)
        delta_inner = sum(np.linalg.norm(B.W[k] - w_old[k]) for k in range(constants.K)) \
                    + sum(np.linalg.norm(B.V[k] - v_old[k]) for k in range(constants.K))
        
        print(f"    Δ_inner = {delta_inner:.6e}")

        if delta_inner < inner_tol:
            print(f"✅ Micro-loop converged at iteration {micro_iter+1}.")
            break

    print("✅ Finished w,v alternating updates.")

    plt.figure()
    plt.plot(range(1, len(v_power_evolution)+1), v_power_evolution)
    plt.xlabel('Micro-Iteration')
    plt.ylabel('Signal Power (||v_n||^2)')
    plt.title('Signal Power Evolution over Micro-Iterations')
    plt.grid(True)
    plt.legend()
    plt.savefig("transmit power")
if __name__ == "__main__":
    test_update_w_v_loop()
