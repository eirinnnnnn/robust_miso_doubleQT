from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import * 
import numpy as np
from scipy.io import loadmat
def test_update_B_loop():
    Delta_k = loadmat("Delta_k.mat")
    Delta_k = Delta_k["Delta_k"] 
    H = loadmat("H_HAT.mat") 
    H = H['H_HAT']
    # === Setup constants and variables
    h_hat_id=3
    constants = GlobalConstants(snr_db=0, snrest_db=10, Nt=32, Nr=2, K=2, Pt=1, Pin=0.75, H_HAT=H[h_hat_id], h_hat_id=h_hat_id)

    A_robust = VariablesA(constants, 0)
    B_robust = VariablesB(constants)

    # === First solve A inner loop (initialize t)
    B_robust = initialize_t(A_robust, B_robust, constants)

    # === Run Algorithm 2 (B update)
    print("=== ROBUST ===")
    # B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2 = update_B_loop_robust(
    # B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2, _,_, v_traj= modified_update_B_loop_robust(
    # B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2= update_B_loop_robust_maxmin(
    B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2= update_B_loop_robust_stableB(
        A_robust, B_robust, constants, max_outer_iter=5000, outer_tol=1e-6, max_inner_iter=2000,inner_tol=1e-4, robust=True
    )


    print("=== NONROBUST ===")
    A_nonrobust = VariablesA(constants)
    B_nonrobust = VariablesB(constants)
    B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
    # B_nonrobust, non_robust_lag,_,_,_,_,_ = update_B_loop_robust_maxmin(A_nonrobust, B_nonrobust, constants, 
    B_nonrobust, non_robust_lag,_,_,_,_,_ = update_B_loop_robust_stableB(A_nonrobust, B_nonrobust, constants, 
    # B_nonrobust, non_robust_lag,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_nonrobust, B_nonrobust, constants, 
                                        max_outer_iter=2000, outer_tol=1e-4, 
                                        max_inner_iter=1000, inner_tol=1e-3, 
                                        robust=False)

    V_wmmse = wmmse_sum_rate(constants, max_iter=1000, tol=1e-4)
    B_wmmse = VariablesB(constants)
    B_wmmse.V = V_wmmse
    # Rate calculation
    r_m, r_v, r_o = compute_rate_test(A_robust, B_robust, constants, Delta_k, samp=5000)
    n_m, n_v, n_o = compute_rate_test(A_nonrobust, B_nonrobust, constants, Delta_k, samp=5000)
    w_m, w_v, w_o = compute_rate_test(A_nonrobust, B_wmmse, constants, Delta_k, samp=5000)

    print(f"mean: robust={r_m:.6e}, non={n_m:.6e}, wmmse={w_m:.6e}")
    print(f"var: robust={r_v:.6e}, non={n_v:.6e}, wmmse={w_v:.6e}")
    print(f"outage rate: robust={r_o:.6e}, non={n_o:.6e}, wmmse={w_o:.6e}")

    # === KKT Condition Check ===
    # alpha_final = alpha_trajectory[-1]
    beta_final = beta_trajectory[-1]

    g_sum = 0
    for k in range(constants.K):
        g_k = compute_rate_k(A_robust, B_robust, constants, k)
        g_sum += g_k

    slack_g = g_sum - B_robust.t
    slack_pwr = constants.PT-sum(np.linalg.norm(B_robust.V[k]) ** 2 for k in range(constants.K))

    # kkt1 = alpha_final * slack_g
    kkt2 = beta_final * slack_pwr

    # print(f"[KKT Check] alpha = {alpha_final:.4e}, constr = {slack_g}, alpha ⋅ (∑g_k - t) = {kkt1:.4e}")
    print(f"[KKT Check] beta = {beta_final:.4e}, constr = {slack_pwr}, β ⋅ (P_T - ∑||v_k||²) = {kkt2:.4e}")

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    
    plt.figure() 
    plt.plot(range(1, len(lagrangian_B_trajectory)+1-100), lagrangian_B_trajectory[100:] )
    plt.xlabel('Outer iteration')
    plt.ylabel('Lagrangian Value')
    plt.title('Algorithm 2 (B Update) Lagrangian Trajectory')
    plt.grid(True)
    plt.savefig("algo2_L_robust.png")

    plt.figure() 
    plt.plot(range(1, len(non_robust_lag)+1-100), non_robust_lag[100:] )
    plt.xlabel('Outer iteration')
    plt.ylabel('Lagrangian Value')
    plt.title('Algorithm 2 Non-robust Design Lagrangian Trajectory')
    plt.grid(True)
    plt.savefig("algo2_L_nonrobust.png")

    # plt.figure() 
    # plt.plot(range(1, len(alpha_trajectory)+1), alpha_trajectory)
    # plt.xlabel('Outer iteration')
    # plt.ylabel('Alpha')
    # plt.title('Algorithm 2 (B Update) ALPHA trajectory')
    # plt.grid(True)
    # ax = plt.gca()  # get current axis
    # # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # # ax.ticklabel_format(style='plain', axis='y')
    # plt.savefig("algo2_alpha.png")

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

    # plt.figure() 
    # plt.plot(range(1, len(t_trajectory)+1), t_trajectory)
    # plt.xlabel('Outer iteration')
    # plt.ylabel('t')
    # plt.title('Algorithm 2 (B Update) t trajectory')
    # plt.grid(True)
    # ax = plt.gca()  # get current axis
    # # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # # ax.ticklabel_format(style='plain', axis='y')
    # plt.savefig("algo2_t.png")
    
    # plt.figure() 
    # plt.plot(range(1, len(res1)+1), res1)
    # plt.xlabel('Outer iteration')
    # plt.ylabel('res')
    # plt.title('Algorithm 2 (B Update) residual 1 trajectory')
    # plt.grid(True)
    # ax = plt.gca()  # get current axis
    # # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # # ax.ticklabel_format(style='plain', axis='y')
    # plt.savefig("algo2_res1.png")
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

    # plt.figure() 
    # plt.plot(range(1, len(v_traj)+1), v_traj)
    # plt.xlabel('Outer iteration')
    # plt.ylabel('res')
    # plt.title('Algorithm 2 (B Update) v_norm trajectory')
    # plt.grid(True)
    # ax = plt.gca()  # get current axis
    # ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    # ax.ticklabel_format(style='plain', axis='y')
    # plt.savefig("algo2_v_norm.png")

if __name__ == "__main__":
    test_update_B_loop()