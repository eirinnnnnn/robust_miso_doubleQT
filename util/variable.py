import numpy as np
from functions import compute_g1_k, generate_delta_within_ellipsoid
from scipy.stats import chi2

class GlobalConstants:
    def __init__(self, snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16):
        self.NT = Nt
        self.NR = Nr
        self.K = K
        self.PT = Pt

        self.SNR_DB = snr_db
        self.SIGMA = 10 ** (-snr_db / 20)

        self.SNREST_DB = snrest_db
        self.SIGMAEST = 10 ** (-snrest_db / 20)

        r2 = chi2.ppf(0.9, df=Nt * Nr * 2)
        self.eps = (1 + 1 / self.SIGMAEST**2) / r2
        self.B = self.eps * np.eye(Nt * Nr)
        self.Binv = np.eye(Nt * Nr) / self.eps

        # Generate K estimated channels \hat{H}_k
        self.H_HAT = np.array([
            np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt)) +
            1j * np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt))
            for _ in range(K)
        ])

        print(f"self.eps: {self.eps:.6f}")

class VariablesA:
    def __init__(self, constants: GlobalConstants):
        self.y = [np.random.randn(constants.NR, 1) for _ in range(constants.K)]
        self.Delta = []
        self.delta = []
        for _ in range(constants.K):
            Delta_k = generate_delta_within_ellipsoid(constants.NR, constants.NT, constants.B)
            self.Delta.append(Delta_k)
            self.delta.append(Delta_k.reshape(-1, 1))

class VariablesB:
    def __init__(self, constants: GlobalConstants):
        self.LAMB = np.ones(constants.K)
        self.ALPHA = 0.1
        self.BETA = 1.0
        self.t = 0.0

        self.W = [
            np.random.randn(constants.NR, 1) + 1j * np.random.randn(constants.NR, 1)
            for _ in range(constants.K)
        ]

        # Initialize V with random complex Gaussian precoders and normalize total power
        self.V = []
        total_power = 0
        for _ in range(constants.K):
            V_k = np.random.normal(0, 1 / np.sqrt(2), (constants.NT, 1)) + \
                  1j * np.random.normal(0, 1 / np.sqrt(2), (constants.NT, 1))
            self.V.append(V_k)
            total_power += np.linalg.norm(V_k) ** 2

        scaling_factor = np.sqrt(constants.PT / total_power)
        self.V = [scaling_factor * V_k for V_k in self.V]

def initialize_t(A: VariablesA, B: VariablesB, constants: GlobalConstants):
    """
    Initialize the slack variable t to be feasible, i.e., t ≤ ∑ g_{1,k}.
    """
    total_g1 = 0
    for k in range(constants.K):
        delta_k = A.delta[k].reshape(constants.NR, constants.NT)
        g1_k = compute_g1_k(A, B, constants, k)

        # Penalty from delta norm constraint
        penalty = (
            delta_k.reshape(-1, 1).conj().T @ constants.B @ delta_k.reshape(-1, 1)
        ).real.item() - 1
        g1_k += B.LAMB[k] * penalty
        total_g1 += g1_k

    B.t = 0.5 * total_g1
    print(f"Initialized t to {B.t:.6e} to ensure feasibility.")
    return B
