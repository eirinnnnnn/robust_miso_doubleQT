from init_class import GlobalConstants, VariablesA, VariablesB
from update_functions import update_A, update_B
from convergence_checker import has_converged

def optimize(
    snr_db=10, snrest_db=11,
    Nt=4, Nr=4, K=4, Pt=16,
    outer_tol=1e-3, max_outer_iter=100
):
    # === Initialization ===
    constants = GlobalConstants(snr_db, snrest_db, Nt, Nr, K, Pt)
    A = VariablesA(constants)
    B = VariablesB(constants)

    for outer_iter in range(max_outer_iter):
        # === Algorithm 1: Update A given fixed B ===
        prev_metric_A = float('inf')
        while True:
            A, metric_A = update_A(A, B, constants)
            if has_converged(prev_metric_A, metric_A):
                break
            prev_metric_A = metric_A

        # === Algorithm 2: Update B given fixed A ===
        prev_metric_B = float('inf')
        while True:
            B, metric_B = update_B(B, A, constants)
            if has_converged(prev_metric_B, metric_B):
                break
            prev_metric_B = metric_B

        # === Outer loop convergence check ===
        if has_converged(metric_A, metric_B, threshold=outer_tol):
            print(f"Optimization converged at outer iteration {outer_iter+1}")
            break

    else:
        print("Maximum outer iterations reached without convergence.")

    return A, B
