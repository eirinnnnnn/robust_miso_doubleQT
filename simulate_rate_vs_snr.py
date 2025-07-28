import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import modified_update_B_loop_robust, update_B_loop_robust, compute_outage_rate, compute_rate_test, wmmse_sum_rate

# def run_simulation_noabort(snr_db_range, n_realizations=5):
#     robust_rates = []
#     nonrobust_rates = []
#     robust_rates_var = []
#     nonrobust_rates_var = []

#     for snr_db in snr_db_range:
#         avg_robust_rate = 0
#         avg_nonrobust_rate = 0
#         avg_robust_var= 0
#         avg_nonrobust_var= 0

#         print(f"\n>>> SNR = {snr_db} dB <<<")
#         for i in range(n_realizations):
#             print(f"  [{snr_db}dB Realization {i+1}/{n_realizations}]")

#             # constants = GlobalConstants(snr_db=snr_db, snrest_db=11, Nt=16, Nr=8, K=8, Pt=100)
#             # constants = GlobalConstants(snr_db=snr_db, snrest_db=0, Nt=16, Nr=2, K=2, Pt=16*2*2)
#             constants = GlobalConstants(snr_db=0, snrest_db=snr_db, Nt=32, Nr=2, K=2, Pt=1, Pin=0.75, h_hat_id=i)
#             # constants = GlobalConstants(snr_db=snr_db, snrest_db=snr_db, Nt=16, Nr=2, K=2, Pt=100)
            
#             # Robust setup
#             A_robust = VariablesA(constants)
#             B_robust = VariablesB(constants)
#             B_robust = initialize_t(A_robust, B_robust, constants)
#             B_robust, _,_,__,_,_,_,_,_,_ = modified_update_B_loop_robust(
#                                             A_robust, B_robust, constants, 
#                                             max_outer_iter=2000, outer_tol=1e-3, 
#                                             max_inner_iter=1000, inner_tol=1e-3, 
#                                             robust=True)

#             # Non-robust setup
#             A_nonrobust = VariablesA(constants)
#             B_nonrobust = VariablesB(constants)
#             B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
#             B_nonrobust, _,_,_,_,_,_ = update_B_loop_robust(A_robust, B_robust, constants, 
#                                                max_outer_iter=2000, outer_tol=1e-3, 
#                                                max_inner_iter=1000, inner_tol=1e-3, 
#                                                robust=False)

#             # robust_rate = sum(compute_rate_k(A_robust, B_robust, constants, k) for k in range(constants.K))
#             # nonrobust_rate = sum(compute_rate_k(A_nonrobust, B_nonrobust, constants, k) for k in range(constants.K))
#             r_m, r_v = compute_rate_test(A_robust, B_robust, constants) 
#             n_m, n_v = compute_rate_test(A_nonrobust, B_nonrobust, constants) 
#             avg_robust_rate += r_m
#             avg_robust_var += r_v 
#             avg_nonrobust_rate += n_m
#             avg_nonrobust_var += n_v 
#             # avg_outage_robust = compute_outage_rate(A_robust, B_robust, constants)

#         robust_rates.append(avg_robust_rate / n_realizations)
#         robust_rates_var.append(avg_robust_var / n_realizations)
#         nonrobust_rates.append(avg_nonrobust_rate / n_realizations)
#         nonrobust_rates_var.append(avg_nonrobust_var / n_realizations)

#     return robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var
from scipy.io import loadmat, savemat
def run_simulation(snr_db_range, n_realizations=5):
    robust_rates = []
    nonrobust_rates = []
    wmmse_rates = []
    robust_rates_var = []
    nonrobust_rates_var = []
    wmmse_rates_var = []


    H = loadmat("H_HAT.mat") 
    H = H['H_HAT']

    Delta_k = loadmat("Delta_k.mat")
    Delta_k = Delta_k["Delta_k"] 
    for snr_db in snr_db_range:
        avg_robust_rate = 0
        avg_nonrobust_rate = 0
        avg_wmmse_rate = 0
        avg_robust_var = 0
        avg_nonrobust_var = 0
        avg_wmmse_var = 0

        n_valid = 0
        n_abort = 0

        print(f"\n>>> SNR = {snr_db} dB <<<")
        for i in range(n_realizations):
            # print(f"  [Realization {i+1}/{n_realizations}]")
            print(f"  [{snr_db}dB Realization {i+1}/{n_realizations}]")
            try:
                # print(H[i].shape)
                constants = GlobalConstants(snr_db=snr_db, snrest_db=5, Nt=32, Nr=2, K=2, Pt=1,Pin=0.75, H_HAT = H[i], h_hat_id=i)

                # Robust setup
                A_robust = VariablesA(constants)
                B_robust = VariablesB(constants)
                B_robust = initialize_t(A_robust, B_robust, constants)
                # B_robust, _, _, _ = update_B_loop_robust(A_robust, B_robust, constants,
                #                                          max_outer_iter=2000, outer_tol=1e-3,
                #                                          max_inner_iter=1000, inner_tol=1e-3,
                #                                          robust=True)
                B_robust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_robust, B_robust, constants, 
                                               max_outer_iter=3500, outer_tol=1e-6, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=True)

                # Check for invalid v_n
                for v in B_robust.V:
                    if np.linalg.norm(v) < 1e-8 or not np.all(np.isfinite(v)):
                        raise ValueError("Invalid v_n in robust design")

                # Non-robust setup
                A_nonrobust = VariablesA(constants)
                B_nonrobust = VariablesB(constants)
                B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
                B_nonrobust, _, _, _,_,_,_ = update_B_loop_robust(A_nonrobust, B_nonrobust, constants,
                                                            max_outer_iter=2000, outer_tol=1e-3,
                                                            max_inner_iter=1000, inner_tol=1e-3,
                                                            robust=False)

                for v in B_nonrobust.V:
                    if np.linalg.norm(v) < 1e-8 or not np.all(np.isfinite(v)):
                        raise ValueError("Invalid v_n in nonrobust design")
                V_wmmse, rate = wmmse_sum_rate(constants, max_iter=1000, tol=1e-4)
                B_wmmse = VariablesB(constants)
                B_wmmse.V = V_wmmse
                # Rate calculation
                r_m, r_v = compute_rate_test(A_robust, B_robust, constants, Delta_k, samp=1000)
                n_m, n_v = compute_rate_test(A_nonrobust, B_nonrobust, constants, Delta_k, samp=1000)
                w_m, w_v = compute_rate_test(A_nonrobust, B_wmmse, constants, Delta_k, samp=1000)
                print(f"mean: robust={r_m:.6e}, non={n_m:.6e}, wmmse={w_m:.6e}")
                print(f"var: robust={r_v:.6e}, non={n_v:.6e}, wmmse={w_v:.6e}")


                avg_robust_rate += r_m
                avg_robust_var += r_v
                avg_nonrobust_rate += n_m
                avg_nonrobust_var += n_v
                avg_wmmse_rate += w_m
                avg_wmmse_var += w_v
                n_valid += 1

            except Exception as e:
                print(f"  ❌ Aborted realization due to: {e}")
                # n_abort += 1
                i = i-1
                continue

        print(f"✔️ Valid realizations = {n_valid}, ❌ Aborted = {n_abort}")
        if n_valid > 0:
            robust_rates.append(avg_robust_rate / n_valid)
            robust_rates_var.append(avg_robust_var / n_valid)
            nonrobust_rates.append(avg_nonrobust_rate / n_valid)
            nonrobust_rates_var.append(avg_nonrobust_var / n_valid)
            wmmse_rates.append(avg_wmmse_rate/n_valid)
            wmmse_rates_var.append(avg_wmmse_var/n_valid)

            savemat('rates.mat', {'r_m': robust_rates,
                                  'r_v': robust_rates_var,
                                  'n_m': nonrobust_rates,
                                  'n_v': nonrobust_rates_var,
                                  'w_m': wmmse_rates,
                                  'w_v': wmmse_rates_var
                                  })
        else:
            robust_rates.append(0)
            robust_rates_var.append(0)
            nonrobust_rates.append(0)
            nonrobust_rates_var.append(0)
            wmmse_rates.append(0)
            wmmse_rates_var.append(0)

    return robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var

def plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates, wmmse_rates):
    plt.figure()
    print("robust mean: ", robust_rates)
    print("nonrobust mean: ", nonrobust_rates)
    print("wmmse mean: ", wmmse_rates)
    plt.plot(snr_db_range, robust_rates, marker='o', label='Robust Design')
    plt.plot(snr_db_range, nonrobust_rates, marker='s', label='Non-Robust Design')
    plt.plot(snr_db_range, wmmse_rates, marker='o', label='WMMSE')
    plt.xlabel('SNR (dB)')
    # plt.xlabel('SNRest (dB)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    plt.title('Rate vs. SNR')
    # plt.title('Rate vs. SNRest')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean.png')

def plot_rate_vs_snr_var(snr_db_range, robust_rates, nonrobust_rates, wmmse_rates):
    plt.figure()
    print("robust var: ", robust_rates)
    print("nonrobust var: ", nonrobust_rates)
    print("wmmse var: ", wmmse_rates)
    plt.plot(snr_db_range, robust_rates, marker='o', label='Robust Design')
    plt.plot(snr_db_range, nonrobust_rates, marker='s', label='Non-Robust Design')
    plt.plot(snr_db_range, wmmse_rates, marker='o', label='WMMSE')
    plt.xlabel('SNR (dB)')
    # plt.xlabel('SNRest (dB)')
    plt.ylabel('Variance of the Sum Rate')
    # plt.title('Rate vs. SNR')
    plt.title('Variance of the Rate Distribution vs. SNR')
    # plt.title('Variance of the Rate Distribution vs. SNRest')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('var.png')


if __name__ == "__main__":
    snr_db_range = np.arange(-3, 4, 1)
    # snrest_db_range = np.arange(-3, 11, 2)
    robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var, w_m, w_v = run_simulation(snr_db_range, n_realizations=25)
    plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates, w_m)
    plot_rate_vs_snr_var(snr_db_range, robust_rates_var, nonrobust_rates_var, w_v)
