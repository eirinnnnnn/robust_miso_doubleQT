import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import update_B_loop_robust, compute_outage_rate, compute_rate_test

def run_simulation(snr_db_range, n_realizations=5):
    robust_rates = []
    nonrobust_rates = []

    for snr_db in snr_db_range:
        avg_robust_rate = 0
        avg_nonrobust_rate = 0

        print(f"\n>>> SNR = {snr_db} dB <<<")
        for i in range(n_realizations):
            print(f"  [Realization {i+1}/{n_realizations}]")

            # constants = GlobalConstants(snr_db=snr_db, snrest_db=11, Nt=16, Nr=8, K=8, Pt=100)
            # constants = GlobalConstants(snr_db=snr_db, snrest_db=0, Nt=16, Nr=2, K=2, Pt=16*2*2)
            constants = GlobalConstants(snr_db=snr_db, snrest_db=0, Nt=32, Nr=1, K=1, Pt=100)
            # constants = GlobalConstants(snr_db=snr_db, snrest_db=snr_db, Nt=16, Nr=2, K=2, Pt=100)
            
            # Robust setup
            A_robust = VariablesA(constants)
            B_robust = VariablesB(constants)
            B_robust = initialize_t(A_robust, B_robust, constants)
            B_robust, _ = update_B_loop_robust(A_robust, B_robust, constants, 
                                               max_outer_iter=500, outer_tol=1e-2, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=True)

            # Non-robust setup
            A_nonrobust = VariablesA(constants)
            B_nonrobust = VariablesB(constants)
            B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
            B_nonrobust, _ = update_B_loop_robust(A_robust, B_robust, constants, 
                                               max_outer_iter=500, outer_tol=1e-2, 
                                               max_inner_iter=1000, inner_tol=1e-3, 
                                               robust=False)

            # robust_rate = sum(compute_rate_k(A_robust, B_robust, constants, k) for k in range(constants.K))
            # nonrobust_rate = sum(compute_rate_k(A_nonrobust, B_nonrobust, constants, k) for k in range(constants.K))

            avg_robust_rate += compute_rate_test(A_robust, B_robust, constants) 
            # avg_outage_robust = compute_outage_rate(A_robust, B_robust, constants)
            avg_nonrobust_rate += compute_rate_test(A_nonrobust, B_nonrobust, constants) 

        robust_rates.append(avg_robust_rate / n_realizations)
        nonrobust_rates.append(avg_nonrobust_rate / n_realizations)

    return robust_rates, nonrobust_rates

def plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates):
    plt.figure()
    print("robust: ", robust_rates)
    print("nonrobust: ", nonrobust_rates)
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
    plt.savefig('converge_tol_test_K1.png')


if __name__ == "__main__":
    snr_db_range = np.arange(-10, 10, 5)
    # snrest_db_range = np.arange(5, 11, 2)
    robust_rates, nonrobust_rates = run_simulation(snr_db_range, n_realizations=30)
    plot_rate_vs_snr(snr_db_range, robust_rates, nonrobust_rates)
