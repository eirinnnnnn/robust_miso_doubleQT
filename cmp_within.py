
import numpy as np
import matplotlib.pyplot as plt
import copy

from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from update_functions import compute_rate_k, update_A_loop, update_B_loop

def generate_deltas(constants, n_samples=1000):
    deltas_all = []
    for _ in range(n_samples):
        deltas = []
        for _ in range(constants.K):
            delta_k = np.random.normal(0, constants.SIGMAEST / np.sqrt(2), (constants.NR, constants.NT)) + 1j * np.random.normal(0, constants.SIGMAEST / np.sqrt(2), (constants.NR, constants.NT))
            deltas.append(delta_k)
        deltas_all.append(deltas)
    return deltas_all

def evaluate(constants, B_robust, B_nonrobust, delta_list):
    rate_robust_all = []
    rate_nonrobust_all = []

    for deltas in delta_list:
        A_robust = VariablesA(constants)
        A_nonrobust = VariablesA(constants)

        for k in range(constants.K):
            A_robust.delta[k] = deltas[k].reshape(-1, 1)
            A_nonrobust.delta[k] = np.zeros((constants.NT * constants.NR, 1))  # set delta to 0

        rate_robust = np.mean([compute_rate_k(A_robust, B_robust, constants, k) for k in range(constants.K)])
        rate_nonrobust = np.mean([compute_rate_k(A_nonrobust, B_nonrobust, constants, k) for k in range(constants.K)])
        rate_robust_all.append(rate_robust)
        rate_nonrobust_all.append(rate_nonrobust)

    return np.mean(rate_robust_all), np.mean(rate_nonrobust_all)

def main():
    snr_dB_range = [0, 5, 10, 15, 20]
    avg_rate_robust = []
    avg_rate_nonrobust = []

    for snr_db in snr_dB_range:
        print(f"=== Training robust model at SNR = {snr_db} dB ===")
        constants = GlobalConstants(snr_db=snr_db, snrest_db=11)
        A = VariablesA(constants)
        B_robust = VariablesB(constants)

        A, _ = update_A_loop(A, B_robust, constants)
        B_robust = initialize_t(A, B_robust, constants)
        B_robust, _ = update_B_loop(A, B_robust, constants, max_outer_iter=100, outer_tol=1e-5, inner_tol=1e-6)

        # Non-robust: independent design, assuming delta = 0
        B_nonrobust = VariablesB(constants)
        B_nonrobust = initialize_t(A, B_nonrobust, constants)
        B_nonrobust, _ = update_B_loop(A, B_nonrobust, constants, max_outer_iter=100, outer_tol=1e-5, inner_tol=1e-6)

        delta_samples = generate_deltas(constants, n_samples=1000)
        r_robust, r_nonrobust = evaluate(constants, B_robust, B_nonrobust, delta_samples)
        avg_rate_robust.append(r_robust)
        avg_rate_nonrobust.append(r_nonrobust)
        print(f"    Robust avg rate: {r_robust:.3f}, Non-robust avg rate: {r_nonrobust:.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(snr_dB_range, avg_rate_robust, label='Robust Design', marker='o')
    plt.plot(snr_dB_range, avg_rate_nonrobust, label='Non-Robust Design', marker='x')
    plt.xlabel("SNR [dB]")
    plt.ylabel("Average Rate (1000 Δ samples)")
    plt.title("Robust vs Non-Robust Design vs SNR (Estimation SNR fixed = 11 dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rate_vs_snr.png")
    plt.show()

if __name__ == "__main__":
    main()
