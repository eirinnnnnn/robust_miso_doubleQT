import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import modified_update_B_loop_robust, update_B_loop_robust, update_B_loop_robust_stableB, compute_rate_test, wmmse_sum_rate, compute_outage_rate

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

    robust_outage = []
    nonrobust_outage = []
    wmmse_outage = []


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

        avg_robust_outage= 0
        avg_nonrobust_outage= 0
        avg_wmmse_outage= 0

        n_valid = 0
        n_abort = 0
        i = 0
        print(f"\n>>> SNR = {snr_db} dB <<<")
        while i < (n_realizations):
            # print(f"  [Realization {i+1}/{n_realizations}]")
            print(f"  [{snr_db}dB Realization {i+1}/{n_realizations}]")
            try:
                # print(H[i].shape)
                constants = GlobalConstants(snr_db=0, snrest_db=snr_db, Nt=32, Nr=2, K=2, Pt=1,Pin=0.75, H_HAT = H[i], h_hat_id=i)

                # Robust setup
                A_robust = VariablesA(constants, delta_k_id = i)
                B_robust = VariablesB(constants)
                B_robust = initialize_t(A_robust, B_robust, constants)
                # B_robust, _, _, _ = update_B_loop_robust(A_robust, B_robust, constants,
                #                                          max_outer_iter=2000, outer_tol=1e-3,
                #                                          max_inner_iter=1000, inner_tol=1e-3,
                #                                          robust=True)
                # B_robust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_robust, B_robust, constants, 
                B_robust, _,_,_,_,_,_= update_B_loop_robust_stableB(A_robust, B_robust, constants,
                                               max_outer_iter=3500, outer_tol=1e-6, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=True)

                # Check for invalid v_n
                for v in B_robust.V:
                    if np.linalg.norm(v) < 1e-8 or not np.all(np.isfinite(v)):
                        raise ValueError("Invalid v_n in robust design")

                # Non-robust setup
                A_nonrobust = VariablesA(constants, delta_k_id = -1)
                B_nonrobust = VariablesB(constants)
                B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
                # B_nonrobust, _, _, _,_,_,_ = update_B_loop_robust(A_nonrobust, B_nonrobust, constants,
                # B_nonrobust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_nonrobust, B_nonrobust, constants, 
                B_nonrobust, _,_,_,_,_,_= update_B_loop_robust_stableB(A_nonrobust, B_nonrobust, constants,
                                                            max_outer_iter=3000, outer_tol=1e-3,
                                                            max_inner_iter=1000, inner_tol=1e-3,
                                                            robust=False)

                for v in B_nonrobust.V:
                    if np.linalg.norm(v) < 1e-8 or not np.all(np.isfinite(v)):
                        raise ValueError("Invalid v_n in nonrobust design")
                V_wmmse = wmmse_sum_rate(constants, max_iter=1000, tol=1e-5)
                B_wmmse = VariablesB(constants)
                B_wmmse.V = V_wmmse
                # Rate calculation
                r_m, r_v, r_o = compute_rate_test(A_robust, B_robust, constants, Delta_k, samp=5000)
                n_m, n_v, n_o = compute_rate_test(A_nonrobust, B_nonrobust, constants, Delta_k, samp=5000)
                w_m, w_v, w_o = compute_rate_test(A_nonrobust, B_wmmse, constants, Delta_k, samp=5000)
                print(f"mean: robust={r_m:.6e}, non={n_m:.6e}, wmmse={w_m:.6e}")
                print(f"var: robust={r_v:.6e}, non={n_v:.6e}, wmmse={w_v:.6e}")
                print(f"outage rate: robust={r_o:.6e}, non={n_o:.6e}, wmmse={w_o:.6e}")


                avg_robust_rate += r_m
                avg_robust_var += r_v
                avg_robust_outage += r_o 
                avg_nonrobust_rate += n_m
                avg_nonrobust_var += n_v
                avg_nonrobust_outage += n_o 
                avg_wmmse_rate += w_m
                avg_wmmse_var += w_v
                avg_wmmse_outage += w_o
                
                n_valid += 1
                i += 1
            except Exception as e:
                print(f"  ❌ Aborted realization due to: {e}")
                n_abort += 1
                # i = i-1

        print(f"✔️ Valid realizations = {n_valid}, ❌ Aborted = {n_abort}")
        if n_valid > 0:
            robust_rates.append(avg_robust_rate / n_valid)
            robust_rates_var.append(avg_robust_var / n_valid)
            robust_outage.append(avg_robust_outage/n_valid)
            nonrobust_rates.append(avg_nonrobust_rate / n_valid)
            nonrobust_rates_var.append(avg_nonrobust_var / n_valid)
            nonrobust_outage.append(avg_nonrobust_outage/n_valid)
            wmmse_rates.append(avg_wmmse_rate/n_valid)
            wmmse_rates_var.append(avg_wmmse_var/n_valid)
            wmmse_outage.append(avg_wmmse_outage/n_valid)

            savemat('rates.mat', {'r_m': robust_rates,
                                  'r_v': robust_rates_var,
                                  'n_m': nonrobust_rates,
                                  'n_v': nonrobust_rates_var,
                                  'w_m': wmmse_rates,
                                  'w_v': wmmse_rates_var,
                                  'r_o': robust_outage,
                                  'n_o': nonrobust_outage,
                                  'w_o': wmmse_outage
                                  })
        else:
            robust_rates.append(0)
            robust_rates_var.append(0)
            nonrobust_rates.append(0)
            nonrobust_rates_var.append(0)
            wmmse_rates.append(0)
            wmmse_rates_var.append(0)

    return robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var, wmmse_rates, wmmse_rates_var, robust_outage, nonrobust_outage, wmmse_outage 



if __name__ == "__main__":
    snr_db_range = np.arange(0, 10, 2)
    # snrest_db_range = np.arange(-3, 11, 2)
    r_m, n_m, r_v, n_v, w_m, w_v, r_o, n_o, w_o = run_simulation(snr_db_range, n_realizations=100)
    print("Robust mean rates (r_m):", r_m)
    print("Robust rate variances (r_v):", r_v)
    print("Non-robust mean rates (n_m):", n_m)
    print("Non-robust rate variances (n_v):", n_v)
    print("WMMSE mean rates (w_m):", w_m)
    print("WMMSE rate variances (w_v):", w_v)
    print("Robust outage rates (r_o):", r_o)
    print("Non-robust outage rates (n_o):", n_o)
    print("WMMSE outage rates (w_o):", w_o)
