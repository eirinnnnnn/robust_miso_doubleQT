import numpy as np

class GlobalConstants:
    def __init__(self, snr_db=10, snrest_db=11, Nt=4, Nr=4, K=4, Pt=16):
        self.NT = Nt
        self.NR = Nr
        self.K = K
        self.PT = Pt

        self.SNR_DB = snr_db
        self.SIGMA = (10 ** (-snr_db / 20))

        self.SNREST_DB = snrest_db
        self.SIGMAEST = (10 ** (-snrest_db / 20))

        # Generate K channel realizations
        self.H_HAT = []
        for _ in range(K):
            h_real = np.random.normal(0, 1/np.sqrt(2), (Nr, Nt))
            h_imag = np.random.normal(0, 1/np.sqrt(2), (Nr, Nt))
            self.H_HAT.append(h_real + 1j * h_imag)
        self.H_HAT = np.array(self.H_HAT)  # shape (K, Nr, Nt)

        # Generate B matrix (identity scaled by sigma estimate)
        self.B = np.eye(Nt * Nr) / self.SIGMAEST**2

class VariablesA:
    def __init__(self, constants: GlobalConstants):
        self.y = [np.random.randn(constants.NR, 1) for _ in range(constants.K)]  # or whatever shape your y_k is


        self.Delta = []
        self.delta = []

        for _ in range(constants.K):
            Delta_k = np.random.normal(0, constants.SIGMAEST/np.sqrt(2), (constants.NR, constants.NT)) \
                      + 1j * np.random.normal(0, constants.SIGMAEST/np.sqrt(2), (constants.NR, constants.NT))
            self.Delta.append(Delta_k)
            self.delta.append(Delta_k.reshape(-1, 1))  # flatten into column

class VariablesB:
    def __init__(self, constants: GlobalConstants):
        self.LAMB = np.ones(constants.K)
        self.ALPHA = 0.5
        self.BETA = 0.5
        self.t = 0

        self.V = []
        total_power = 0

        for _ in range(constants.K):
            V_k = np.random.normal(0, 1/np.sqrt(2), (constants.NT, 1)) \
                  + 1j * np.random.normal(0, 1/np.sqrt(2), (constants.NT, 1))
            self.V.append(V_k)
            total_power += np.linalg.norm(V_k**2) 
            # print(V_k, total_power)

        scaling_factor = np.sqrt(constants.PT / total_power)
        self.V = [scaling_factor * V_k for V_k in self.V]
