from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_w, update_v
import numpy as np
def test_update_w_v():
    # === Step 1: Initialize
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16)
    A = VariablesA(constants)
    B = VariablesB(constants)

    print("=== Testing update_w and update_v ===")

    # === Step 2: Update w
    for k in range(constants.K):
        w_k = update_w(A, B, constants, k)
        B.W[k] = w_k
        w_norm = np.linalg.norm(w_k)
        print(f"[w_{k}] norm = {w_norm:.6e}")
        assert np.isfinite(w_norm), f"w_{k} has non-finite norm!"

    # === Step 3: Update v
    for n in range(constants.K):
        v_n = update_v(A, B, constants, n)
        B.V[n] = v_n
        v_norm = np.linalg.norm(v_n)
        print(f"[v_{n}] norm = {v_norm:.6e}")
        assert np.isfinite(v_norm), f"v_{n} has non-finite norm!"

    print("✅ All w_k and v_n updated successfully with finite norms.")

if __name__ == "__main__":
    test_update_w_v()
