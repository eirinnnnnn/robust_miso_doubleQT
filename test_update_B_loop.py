from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from update_functions import update_A_loop, update_B_loop

def test_update_B_loop():
    # === Setup constants and variables
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=8, Nr=8, K=8, Pt=64)
    A = VariablesA(constants)
    B = VariablesB(constants)

    # === First solve A inner loop (optional if you want fresh delta/y)
    A, converged_A= update_A_loop(A, B, constants)
    B = initialize_t(A, B, constants)
    # === Now test B inner loop
    print("=== Starting Algorithm 2 (B update) ===")
    B, _, _ = update_B_loop(A, B, constants)

    print("=== Finished Algorithm 2 ===")
    print(f"Final outer Lagrangian: {lagrangian_B_trajectory[-1]:.6e}")

    # (Optional: plot Lagrangian curve here if you want)
    import matplotlib.pyplot as plt
    plt.figure() 

    plt.plot(range(1, len(lagrangian_B_trajectory)+1), lagrangian_B_trajectory )
    plt.xlabel('Outer iteration')
    plt.ylabel('Lagrangian Value')
    plt.title('Algorithm 2 (B Update) Lagrangian Trajectory')
    plt.grid(True)
    plt.savefig("algo2_L.png")

if __name__ == "__main__":
    test_update_B_loop()
