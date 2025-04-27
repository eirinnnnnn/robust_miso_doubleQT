from variables import GlobalConstants, VariablesA, VariablesB
import numpy as np

def test_variables():
    print("Testing variables.py ...")

    # Example config
    snr_db = 10
    snrest_db = 11
    Nt = 4
    Nr = 4
    K = 4
    Pt = 16

    # Initialize
    constants = GlobalConstants(snr_db, snrest_db, Nt, Nr, K, Pt)
    A = VariablesA(constants)
    B = VariablesB(constants)

    # === Test GlobalConstants ===
    print("\nTesting GlobalConstants...")
    assert constants.H_HAT.shape == (K, Nr, Nt), f"Unexpected H_HAT shape: {constants.H_HAT.shape}"
    assert constants.B.shape == (Nt*Nr, Nt*Nr), f"Unexpected B shape: {constants.B.shape}"
    print("GlobalConstants looks good.")

    # === Test VariablesA ===
    print("\nTesting VariablesA...")
    assert len(A.Delta) == K, f"Expected {K} Delta matrices, got {len(A.Delta)}"
    assert len(A.delta) == K, f"Expected {K} delta vectors, got {len(A.delta)}"
    for idx in range(K):
        assert A.Delta[idx].shape == (Nr, Nt), f"Delta[{idx}] wrong shape: {A.Delta[idx].shape}"
        assert A.delta[idx].shape == (Nr*Nt, 1), f"delta[{idx}] wrong shape: {A.delta[idx].shape}"
    print("VariablesA looks good.")

    # === Test VariablesB ===
    print("\nTesting VariablesB...")
    assert len(B.V) == K, f"Expected {K} precoders, got {len(B.V)}"
    total_power = sum(np.linalg.norm(Vk**2) for Vk in B.V)
    print(f"Total precoding power = {total_power:.4f} (target {Pt})")
    assert np.isclose(total_power, Pt, atol=1e-2), f"Total power {total_power} not close to {Pt}"

    print(f"LAMB shape: {B.LAMB.shape}")
    assert B.LAMB.shape == (K,), f"LAMB shape wrong: {B.LAMB.shape}"
    print(f"ALPHA: {B.ALPHA}, BETA: {B.BETA}, t: {B.t}")

    print("VariablesB looks good.")

    print("\n✅ All tests passed.")

if __name__ == "__main__":
    test_variables()
