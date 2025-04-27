from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_A, update_A_loop

def test_update_A():
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16)
    A = VariablesA(constants)
    B = VariablesB(constants)

    print("Initial metric:")
    _, metric = update_A(A, B, constants)
    print(f"Metric after one inner iteration: {metric}")

if __name__ == "__main__":
    test_update_A()
from variables import GlobalConstants, VariablesA, VariablesB
from update_functions import update_A

def test_update_A_loop():
    constants = GlobalConstants(snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16)
    A = VariablesA(constants)
    B = VariablesB(constants)

    print("Starting Algorithm 1 inner loop...")
    A = update_A_loop(A, B, constants)

if __name__ == "__main__":
    test_update_A_loop()
