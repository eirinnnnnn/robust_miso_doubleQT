from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import modified_update_B_loop_robust, update_B_loop_robust_stableB,update_B_loop_robust, compute_rate_test, compute_rate_over, compute_g1_k_QT
import numpy as np
def test_update_B_loop():
    # === Setup constants and variables
    constants = GlobalConstants(snr_db=0, snrest_db=5, Nt=32, Nr=2, K=2, Pt=4)
    A_robust = VariablesA(constants)
    B_robust = VariablesB(constants)

    # === First solve A inner loop (initialize t)
    B_robust = initialize_t(A_robust, B_robust, constants)

    # === Run Algorithm 2 (B update)
    print("=== ROBUST ===")
    B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2, _,_, v_traj= modified_update_B_loop_robust(
        A_robust, B_robust, constants, max_outer_iter=1000, outer_tol=1e-4, inner_tol=1e-3, robust=True
    )

    # === Evaluate final rate
    robust_rate_in, rob_var = compute_rate_test(A_robust, B_robust, constants, samp=10000)
    robust_rate_over = compute_rate_over(A_robust, B_robust, constants)

    print("=== NONROBUST ===")
    A_nonrobust = VariablesA(constants)
    B_nonrobust = VariablesB(constants)
    B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
    B_nonrobust, _,_,_,_,_,_ = update_B_loop_robust(A_robust, B_robust, constants, 
                                        max_outer_iter=500, outer_tol=5e-3, 
                                        max_inner_iter=1000, inner_tol=1e-3, 
                                        robust=False)
    nonrobust_rate_in, non_var = compute_rate_test(A_nonrobust, B_nonrobust, constants, samp=10000)
    nonrobust_rate_over = compute_rate_over(A_nonrobust, B_nonrobust, constants)
    print(f"final robust rate: mean={robust_rate_in:.6e}, var={rob_var:.6e}")
    print(f"final robust over region rate: {robust_rate_over:.6e}")
    print(f"final nonrobust rate: mean={nonrobust_rate_in:.6e}, var={non_var:.6e}")
    print(f"final nonrobust over region rate: {nonrobust_rate_over:.6e}")
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

    plt.figure() 
    plt.plot(range(1, len(t_trajectory)+1), t_trajectory)
    plt.xlabel('Outer iteration')
    plt.ylabel('t')
    plt.title('Algorithm 2 (B Update) t trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_t.png")
    
    plt.figure() 
    plt.plot(range(1, len(res1)+1), res1)
    plt.xlabel('Outer iteration')
    plt.ylabel('res')
    plt.title('Algorithm 2 (B Update) residual 1 trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_res1.png")
    plt.figure() 
    plt.plot(range(1, len(res2)+1), res2)
    plt.xlabel('Outer iteration')
    plt.ylabel('res')
    plt.title('Algorithm 2 (B Update) residual 2 trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_res2.png")

    plt.figure() 
    plt.plot(range(1, len(v_traj)+1), v_traj)
    plt.xlabel('Outer iteration')
    plt.ylabel('res')
    plt.title('Algorithm 2 (B Update) v_norm trajectory')
    plt.grid(True)
    ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    plt.savefig("algo2_v_norm.png")

if __name__ == "__main__":
    test_update_B_loop()