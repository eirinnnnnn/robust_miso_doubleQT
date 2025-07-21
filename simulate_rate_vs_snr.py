import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import modified_update_B_loop_robust, update_B_loop_robust, compute_outage_rate, compute_rate_test

def run_simulation_noabort(snr_db_range, n_realizations=5):
    robust_rates = []
    nonrobust_rates = []
    robust_rates_var = []
    nonrobust_rates_var = []

    for snr_db in snr_db_range:
        avg_robust_rate = 0
        avg_nonrobust_rate = 0
        avg_robust_var= 0
        avg_nonrobust_var= 0

        print(f"\n>>> SNR = {snr_db} dB <<<")
        for i in range(n_realizations):
            print(f"  [Realization {i+1}/{n_realizations}]")

            # constants = GlobalConstants(snr_db=snr_db, snrest_db=11, Nt=16, Nr=8, K=8, Pt=100)
            # constants = GlobalConstants(snr_db=snr_db, snrest_db=0, Nt=16, Nr=2, K=2, Pt=16*2*2)
            constants = GlobalConstants(snr_db=0, snrest_db=snr_db, Nt=32, Nr=2, K=2, Pt=4)
            # constants = GlobalConstants(snr_db=snr_db, snrest_db=snr_db, Nt=16, Nr=2, K=2, Pt=100)
            
            # Robust setup
            A_robust = VariablesA(constants)
            B_robust = VariablesB(constants)
            B_robust = initialize_t(A_robust, B_robust, constants)
            B_robust, _,_,__,_,_,_,_,_,_ = modified_update_B_loop_robust(
                                            A_robust, B_robust, constants, 
                                            max_outer_iter=2000, outer_tol=1e-3, 
                                            max_inner_iter=1000, inner_tol=1e-3, 
                                            robust=True)

            # Non-robust setup
            A_nonrobust = VariablesA(constants)
            B_nonrobust = VariablesB(constants)
            B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
            B_nonrobust, _,_,_,_,_,_ = update_B_loop_robust(A_robust, B_robust, constants, 
                                               max_outer_iter=2000, outer_tol=1e-3, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=False)

            # robust_rate = sum(compute_rate_k(A_robust, B_robust, constants, k) for k in range(constants.K))
            # nonrobust_rate = sum(compute_rate_k(A_nonrobust, B_nonrobust, constants, k) for k in range(constants.K))
            r_m, r_v = compute_rate_test(A_robust, B_robust, constants) 
            n_m, n_v = compute_rate_test(A_nonrobust, B_nonrobust, constants) 
            avg_robust_rate += r_m
            avg_robust_var += r_v 
            avg_nonrobust_rate += n_m
            avg_nonrobust_var += n_v 
            # avg_outage_robust = compute_outage_rate(A_robust, B_robust, constants)

        robust_rates.append(avg_robust_rate / n_realizations)
        robust_rates_var.append(avg_robust_var / n_realizations)
        nonrobust_rates.append(avg_nonrobust_rate / n_realizations)
        nonrobust_rates_var.append(avg_nonrobust_var / n_realizations)

    return robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var
def run_simulation(snr_db_range, n_realizations=5):
    robust_rates = []
    nonrobust_rates = []
    robust_rates_var = []
    nonrobust_rates_var = []

    for snr_db in snr_db_range:
        avg_robust_rate = 0
        avg_nonrobust_rate = 0
        avg_robust_var = 0
        avg_nonrobust_var = 0
        n_valid = 0
        n_abort = 0

        print(f"\n>>> SNR = {snr_db} dB <<<")
        for i in range(n_realizations):
            print(f"  [Realization {i+1}/{n_realizations}]")
            try:
                constants = GlobalConstants(snr_db=0, snrest_db=snr_db, Nt=32, Nr=2, K=2, Pt=1)

                # Robust setup
                A_robust = VariablesA(constants)
                B_robust = VariablesB(constants)
                B_robust = initialize_t(A_robust, B_robust, constants)
                # B_robust, _, _, _ = update_B_loop_robust(A_robust, B_robust, constants,
                #                                          max_outer_iter=2000, outer_tol=1e-3,
                #                                          max_inner_iter=1000, inner_tol=1e-3,
                #                                          robust=True)
                B_robust, _,_,_,_,_,_,_,_,_ = modified_update_B_loop_robust(A_robust, B_robust, constants, 
                                               max_outer_iter=2000, outer_tol=5e-4, 
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

                # Rate calculation
                r_m, r_v = compute_rate_test(A_robust, B_robust, constants)
                n_m, n_v = compute_rate_test(A_nonrobust, B_nonrobust, constants)

                avg_robust_rate += r_m
                avg_robust_var += r_v
                avg_nonrobust_rate += n_m
                avg_nonrobust_var += n_v
                n_valid += 1

            except Exception as e:
                print(f"  ❌ Aborted realization due to: {e}")
                n_abort += 1
                continue

        print(f"✔️ Valid realizations = {n_valid}, ❌ Aborted = {n_abort}")
        if n_valid > 0:
            robust_rates.append(avg_robust_rate / n_valid)
            robust_rates_var.append(avg_robust_var / n_valid)
            nonrobust_rates.append(avg_nonrobust_rate / n_valid)
            nonrobust_rates_var.append(avg_nonrobust_var / n_valid)
        else:
            robust_rates.append(0)
            robust_rates_var.append(0)
            nonrobust_rates.append(0)
            nonrobust_rates_var.append(0)

    return robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var

def plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates):
    plt.figure()
    print("robust mean: ", robust_rates)
    print("nonrobust mean: ", nonrobust_rates)
    plt.plot(snr_db_range, robust_rates, marker='o', label='Robust Design')
    plt.plot(snr_db_range, nonrobust_rates, marker='s', label='Non-Robust Design')
    # plt.xlabel('SNR (dB)')
    plt.xlabel('SNRest (dB)')
    plt.ylabel('Average Sum Rate (bps/Hz)')
    # plt.title('Rate vs. SNR')
    plt.title('Rate vs. SNRest')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('mean.png')

def plot_rate_vs_snr_var(snr_db_range, robust_rates, nonrobust_rates):
    plt.figure()
    print("robust var: ", robust_rates)
    print("nonrobust var: ", nonrobust_rates)
    plt.plot(snr_db_range, robust_rates, marker='o', label='Robust Design')
    plt.plot(snr_db_range, nonrobust_rates, marker='s', label='Non-Robust Design')
    # plt.xlabel('SNR (dB)')
    plt.xlabel('SNRest (dB)')
    plt.ylabel('Variance of the Sum Rate')
    # plt.title('Rate vs. SNR')
    plt.title('Variance of the Rate Distribution vs. SNRest')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('var.png')


if __name__ == "__main__":
    snr_db_range = np.arange(-5, 6, 1)
    # snrest_db_range = np.arange(5, 11, 2)
    robust_rates, nonrobust_rates, robust_rates_var, nonrobust_rates_var = run_simulation(snr_db_range, n_realizations=200)
    plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates)
    plot_rate_vs_snr_var(snr_db_range, robust_rates_var, nonrobust_rates_var)
