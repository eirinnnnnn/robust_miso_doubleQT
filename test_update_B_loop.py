from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from functions import update_B_loop_robust, compute_rate_test, compute_rate_over

def test_update_B_loop():
    # === Setup constants and variables
    constants = GlobalConstants(snr_db=5, snrest_db=10, Nt=8, Nr=2, K=2, Pt=100)
    A_robust = VariablesA(constants)
    B_robust = VariablesB(constants)

    # A_nonrobust = VariablesA(constants)
    # B_nonrobust = VariablesB(constants)
    # === First solve A inner loop (optional if you want fresh delta/y)
    B_robust = initialize_t(A_robust, B_robust, constants)
    # B_nonrobust = initialize_t(A_nonrobust, B_nonrobust, constants)
    # === Now test B inner loop
    print("=== Starting Algorithm 2 (B update) ===")
    B_robust, lagrangian_B_trajectory, alpha_trajectory, beta_trajectory = update_B_loop_robust(A_robust, B_robust, constants, max_outer_iter=5000, outer_tol=1e-4, inner_tol=1e-3, robust=True)
    # B_nonrobust, _ = update_B_loop_robust(A_nonrobust, B_nonrobust, constants, max_outer_iter=5000, outer_tol=1e-4, robust=False)

    print("=== Finished Algorithm 2 ===")

    robust_rate = compute_rate_test(A_robust, B_robust, constants) 
    # nonrobust_rate = compute_rate_test(A_nonrobust, B_nonrobust, constants) 
    print(f"final robust rate: {robust_rate:.6e}")
    # print(f"final robust rate: {robust_rate:.6e}, non robust rate: {nonrobust_rate:.6e}")
    # print(f"Final outer Lagrangian: {lagrangian_B_trajectory[-1]:.6e}, final robust rate: {robust_rate:.6e}")
    robust_rate = compute_rate_over(A_robust, B_robust, constants) 
    # nonrobust_rate = compute_rate_over(A_nonrobust, B_nonrobust, constants) 
    print(f"final robust over region rate: {robust_rate:.6e}")
    # print(f"final robust rate: {robust_rate:.6e}, non robust rate: {nonrobust_rate:.6e}")


    # (Optional: plot Lagrangian curve here if you want)
    import matplotlib.pyplot as plt
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
    plt.savefig("algo2_alpha.png")

    plt.figure() 
    plt.plot(range(1, len(beta_trajectory)+1), beta_trajectory)
    plt.xlabel('Outer iteration')
    plt.ylabel('Beta')
    plt.title('Algorithm 2 (B Update) BETA trajectory')
    plt.grid(True)
    plt.savefig("algo2_beta.png")

if __name__ == "__main__":
    test_update_B_loop()
