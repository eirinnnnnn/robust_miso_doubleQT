from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_A, update_A_loop

def test_update_A_loop():
    constants = GlobalConstants(snr_db=10, snrest_db=20, Nt=4, Nr=4, K=4, Pt=100)
    A = VariablesA(constants)
    B = VariablesB(constants)

    print("Starting Algorithm 1 inner loop...")
    A = update_A_loop(A, B, constants, max_inner_iter=30)

if __name__ == "__main__":
    test_update_A_loop()
