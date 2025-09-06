from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import update_loop_learnable, compute_rate_test 
from functions import * 
import numpy as np
from scipy.io import loadmat, savemat

import torch

def run_simulation(snr_db_range, n_realizations=5):
    robust_rates = []
    nonrobust_rates = []
    wmmse_rates = []
    zf_rates = []

    robust_rates_var = []
    nonrobust_rates_var = []
    wmmse_rates_var = []
    zf_rates_var = []

    robust_outage = []
    nonrobust_outage = []
    wmmse_outage = []
    zf_outage = []


    H = loadmat("H_HAT.mat") 
    H = H['H_HAT']

    Delta_k = loadmat("Delta_k.mat")
    Delta_k = Delta_k["Delta_k"] 
    for snr_db in snr_db_range:
        avg_robust_rate = 0
        avg_nonrobust_rate = 0
        avg_wmmse_rate = 0
        avg_zf_rate = 0

        avg_robust_var = 0
        avg_nonrobust_var = 0
        avg_wmmse_var = 0
        avg_zf_var = 0

        avg_robust_outage= 0
        avg_nonrobust_outage= 0
        avg_wmmse_outage= 0
        avg_zf_outage= 0

        n_valid = 0
        n_abort = 0
        i = 0
        print(f"\n>>> SNR = {snr_db} dB <<<")
        while i < (n_realizations):
            # print(f"  [Realization {i+1}/{n_realizations}]")
            print(f"  [{snr_db}dB Realization {i+1}/{n_realizations}]")
            try:
                # print(H[i].shape)
                constants = GlobalConstants(snr_db=5, snrest_db=[snr_db, snr_db], Nt=32, Nr=2, K=2, Pt=2, Pin=0.75, H_HAT=H[i+1], h_hat_id=i+1)
        

                # Robust setup
                A_robust = VariablesA(constants, delta_k_id = i)
                B_robust = VariablesB(constants)
                B_robust = initialize_t(A_robust, B_robust, constants)
                # B_robust, _, _, _ = update_B_loop_robust(A_robust, B_robust, constants,
                #                                          max_outer_iter=2000, outer_tol=1e-3,
                #                                          max_inner_iter=1000, inner_tol=1e-3,
                #                                          robust=True)
                # B_robust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_robust, B_robust, constants, 
                # B_robust, _,_,_,_,_,_= update_B_loop_robust_stableB(A_robust, B_robust, constants,
                B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory , t_trajectory, res1, res2= update_loop_learnable(
                                                A_robust, B_robust, constants,
                                               max_outer_iter=20000, outer_tol=1e-6, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=True)

                # Check for invalid v_n

                # Non-robust setup
                A_nonrobust = VariablesA(constants, delta_k_id = -1)
                B_nonrobust = VariablesB(constants)
                B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
                # B_nonrobust, _, _, _,_,_,_ = update_B_loop_robust(A_nonrobust, B_nonrobust, constants,
                # B_nonrobust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_nonrobust, B_nonrobust, constants, 
                # B_nonrobust, _,_,_,_,_,_= update_B_loop_robust_stableB(
                B_nonrobust, non_robust_lag,_,non_beta_traj,_,_,non_res2_traj = update_loop_learnable(
                                                            A_nonrobust, B_nonrobust, constants,
                                                            max_outer_iter=10000, outer_tol=1e-5,
                                                            max_inner_iter=1000, inner_tol=1e-3,
                                                            robust=False)

                V_wmmse = wmmse_sum_rate(constants, max_iter=1000, tol=1e-5)
                B_wmmse = VariablesB(constants)
                B_wmmse.V = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(v, dtype=torch.cfloat)) for v in V_wmmse
                ])
                V_zf = zf_precoder(constants)
                B_zf = VariablesB(constants)
                B_zf.V = torch.nn.ParameterList([
                torch.nn.Parameter(torch.tensor(v, dtype=torch.cfloat)) for v in V_zf
                ])
                # Rate calculation
                r, r_m, r_v, r_o = compute_rate_test(A_robust, B_robust, constants, Delta_k, snr_db, samp=5000)
                n, n_m, n_v, n_o = compute_rate_test(A_nonrobust, B_nonrobust, constants, Delta_k, snr_db,samp=5000)
                w, w_m, w_v, w_o = compute_rate_test(A_nonrobust, B_wmmse, constants, Delta_k, snr_db,samp=5000)
                z, z_m, z_v, z_o = compute_rate_test(A_nonrobust, B_zf, constants, Delta_k, snr_db, samp=5000)

                print(f"robust={r_m};{r_o}") 
                print(f"non={n_m};{n_o}")
                print(f"wmmse={w_m};{w_o}")
                print(f"zf={z_m};{z_o}")

                avg_robust_rate += np.array(r_m)
                avg_robust_var += np.array(r_v)
                avg_robust_outage += np.array(r_o) 
                avg_nonrobust_rate += np.array(n_m)
                avg_nonrobust_var += np.array(n_v)
                avg_nonrobust_outage += np.array(n_o )
                avg_wmmse_rate += np.array(w_m)
                avg_wmmse_var += np.array(w_v)
                avg_wmmse_outage += np.array(w_o)
                avg_zf_rate += np.array(z_m)
                avg_zf_var += np.array(z_v)
                avg_zf_outage += np.array(z_o)
                
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
            zf_rates.append(avg_zf_rate/n_valid)
            zf_rates_var.append(avg_zf_var/n_valid)
            zf_outage.append(avg_zf_outage/n_valid)

            savemat('rates.mat', {'r_m': robust_rates,
                                  'r_v': robust_rates_var,
                                  'r_o': robust_outage,
                                  'n_m': nonrobust_rates,
                                  'n_v': nonrobust_rates_var,
                                  'n_o': nonrobust_outage,
                                  'w_m': wmmse_rates,
                                  'w_v': wmmse_rates_var,
                                  'w_o': wmmse_outage,
                                  'z_m': zf_rates,
                                  'z_v': zf_rates_var,
                                  'z_o': zf_outage, 
                                  'snr': snr_db_range
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
    snr_db_range = np.arange(-30, 31, 5)
    # snrest_db_range = np.arange(-3, 11, 2)
    r_m, n_m, r_v, n_v, w_m, w_v, r_o, n_o, w_o = run_simulation(snr_db_range, n_realizations=50)
    print("Robust mean rates (r_m):", r_m)
    print("Robust rate variances (r_v):", r_v)
    print("Non-robust mean rates (n_m):", n_m)
    print("Non-robust rate variances (n_v):", n_v)
    print("WMMSE mean rates (w_m):", w_m)
    print("WMMSE rate variances (w_v):", w_v)
    print("Robust outage rates (r_o):", r_o)
    print("Non-robust outage rates (n_o):", n_o)
    print("WMMSE outage rates (w_o):", w_o)
