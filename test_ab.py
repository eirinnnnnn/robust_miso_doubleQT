import numpy as np
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from update_functions import update_A_loop, update_B_loop

# === Define test configuration ===
constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=16, Nr=8, K=8, Pt=100000)  # Use Pt = 10 W
A = VariablesA(constants)
B = VariablesB(constants)
B = initialize_t(A, B, constants)  # Ensure t starts feasible

# === Alternating optimization loop ===
max_iterations = 500
outer_lagrangian_trajectory = []

A, A_converged = update_A_loop(A, B, constants, inner_tol=1e-4, max_inner_iter=500, plot_lagrangian=False)
for iteration in range(max_iterations):
    print(f"\n========== Iteration {iteration + 1} ==========")

    # Step 1: Update A (y_k, delta_k) given B

    # Step 2: Update B (v_k, w_k, lambda_k, alpha, beta, t) given updated A
    B, B_lagrangian_trajectory = update_B_loop(A, B, constants, outer_tol=1e-6, max_outer_iter=1, inner_tol=1e-4, max_inner_iter=500)

    # Track outer objective
    outer_lagrangian_trajectory.append(B_lagrangian_trajectory[-1])

    # Convergence check (optional)
    if iteration > 0:
        delta = abs(outer_lagrangian_trajectory[-1] - outer_lagrangian_trajectory[-2])
        print(f"    Total Lagrangian change = {delta:.6e}")
        if delta < 1e-4:
            print("✅ Overall alternating loop converged.")
            break

# === Final status ===
print("\nFinal Lagrangian trajectory:")
for i, val in enumerate(outer_lagrangian_trajectory):
    print(f"  Iter {i+1}: {val:.6e}")