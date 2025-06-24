from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import update_B_loop_robust, compute_rate_test, compute_rate_over, compute_g1_k_QT
import numpy as np
def test_update_B_loop():
    # === Setup constants and variables
    constants = GlobalConstants(snr_db=5, snrest_db=10, Nt=8, Nr=2, K=2, Pt=100)
    A_robust = VariablesA(constants)
    B_robust = VariablesB(constants)

    # === First solve A inner loop (initialize t)
    B_robust = initialize_t(A_robust, B_robust, constants)

    # === Run Algorithm 2 (B update)
    print("=== Starting Algorithm 2 (B update) ===")
    B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory = update_B_loop_robust(
        A_robust, B_robust, constants, max_outer_iter=5000, outer_tol=1e-4, inner_tol=1e-3, robust=True
    )
    print("=== Finished Algorithm 2 ===")

    # === Evaluate final rate
    robust_rate = compute_rate_test(A_robust, B_robust, constants)
    print(f"final robust rate: {robust_rate:.6e}")
    robust_rate = compute_rate_over(A_robust, B_robust, constants)
    print(f"final robust over region rate: {robust_rate:.6e}")

    # === KKT Condition Check ===
    alpha_final = alpha_trajectory[-1]
    beta_final = beta_trajectory[-1]

    # Compute g_sum = sum_k g_{1,k}(V)
    g_sum = 0
    for k in range(constants.K):
        g_k = compute_g1_k_QT(A_robust, B_robust, constants, k)
        g_sum += g_k

    slack_g = g_sum - B_robust.t
    slack_pwr = sum(np.linalg.norm(B_robust.V[k]) ** 2 for k in range(constants.K)) - constants.PT

    kkt1 = alpha_final * slack_g
    kkt2 = beta_final * slack_pwr

    print(f"[KKT Check] alpha = {alpha_final:.4e}, constr = {slack_g}, alpha ⋅ (∑g_k - t) = {kkt1:.4e}")
    print(f"[KKT Check] beta = {beta_final:.4e}, constr = {slack_pwr}, β ⋅ (∑||v_k||² - P_T) = {kkt2:.4e}")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    plt.figure() 
    plt.plot(range(1, len(lagrangian_B_trajectory)+1), lagrangian_B_trajectory )
    plt.xlabel('Outer iteration')
    plt.ylabel('Lagrangian Value')
    plt.title('Algorithm 2 (B Update) Lagrangian Trajectory')
    plt.grid(True)
    plt.savefig("algo2_L.png")

    plt.figure() 
    plt.plot(range(1, len(alpha_trajectory)+1), alpha_trajectory)
    plt.xlabel('Outer iteration')
    plt.ylabel('Alpha')
    plt.title('Algorithm 2 (B Update) ALPHA trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_alpha.png")

    plt.figure() 
    plt.plot(range(1, len(beta_trajectory)+1), beta_trajectory)
    plt.xlabel('Outer iteration')
    plt.ylabel('Beta')
    plt.title('Algorithm 2 (B Update) BETA trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_beta.png")

if __name__ == "__main__":
    test_update_B_loop()