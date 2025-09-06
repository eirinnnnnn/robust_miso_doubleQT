import numpy as np
from functions import compute_g1_k, generate_delta_within_ellipsoid 
from scipy.stats import chi2
from scipy.io import loadmat
class GlobalConstants:
    def __init__(self, H_HAT, snr_db=10, snrest_db=[11], Nt=4, Nr=4, K=4, Pt=16, Pin=0.9, h_hat_id=-1, ):
        self.NT = Nt
        self.NR = Nr
        self.K = K
        self.PT = Pt

        self.SNR_DB = snr_db
        self.SIGMA = (10 ** (-snr_db / 20))

        r2 = chi2.ppf(Pin, df=Nt*Nr*2)

        # Generate K channel realizations
        # self.H_HAT = []
        # for _ in range(K):
        #     h_real = np.random.normal(0, 1/np.sqrt(2), (Nr, Nt))
        #     h_imag = np.random.normal(0, 1/np.sqrt(2), (Nr, Nt))
        #     self.H_HAT.append(h_real + 1j * h_imag)
        # self.H_HAT = np.array(self.H_HAT)  # shape (K, Nr, Nt)
        self.eps = []
        self.B = []
        self.Binv = []
        self.SIGMAEST = []
        self.SNREST_DB = snrest_db
        # for k in range(K):
        #     self.SIGMAEST.append(10 ** (-snrest_db[k] / 20))
        #     self.eps.append((1 + 1 / self.SIGMAEST[k]**2) / r2) 
        #     self.B.append(self.eps[k] * np.eye(Nt*Nr) )
        #     self.Binv.append(np.eye(Nt*Nr) / self.eps[k])
        
        for k in range(K):
            sigma_est = 10 ** (snrest_db[k] / 20)
            self.SIGMAEST.append(sigma_est)
            eps_k = np.sqrt((1 + 1 / sigma_est**2) * r2)
            self.eps.append(eps_k)
            self.B.append(np.eye(Nt*Nr))
            self.Binv.append(np.eye(Nt*Nr))
        if h_hat_id==-1:
            H_HAT = loadmat("H_HAT.mat") 
            H_HAT = H_HAT['H_HAT'][h_hat_id]

        self.H_HAT = H_HAT
        # for k in range(K):
            # scale = self.SIGMAEST[k]
            # H_HAT[:][k] *= scale 


class VariablesA:
    def __init__(self, constants: GlobalConstants, delta_k_id=-1, robust=True):
        self.y = [np.random.randn(constants.NR, 1) for _ in range(constants.K)]  # or whatever shape your y_k is

        # scale = 1
        self.Delta = loadmat("Delta_k.mat")["Delta_k"][delta_k_id]
        # for k in range(constants.K):
            # scale = np.sqrt(1/ (1 + 10**(constants.SNREST_DB[k]/10)))
            # self.Delta[k] *= scale

        if robust == False:
            self.Delta = np.zeros(shape=self.Delta.shape)

        self.delta = [Delta_k.reshape(-1, 1) for Delta_k in self.Delta]


class VariablesB:
    def __init__(self, constants: GlobalConstants):
        self.LAMB = np.ones(constants.K)
        self.ALPHA = 5 
        self.BETA = 5 
        self.t = 0

        self.W = [np.random.randn(constants.NR, 1) + 1j*np.random.randn(constants.NR, 1) for _ in range(constants.K)]
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
def initialize_t(A, B, constants):
    """
    Initialize t to be feasible (t <= sum_k g1_k).
    """

    K = constants.K
    sigma2 = constants.SIGMA**2

    total_g1 = 0
    for k in range(K):
        Nr = constants.NR
        Nt = constants.NT
        H_hat_k = constants.H_HAT[k]
        delta_k = A.delta[k].reshape(Nr, Nt)
        H_k = H_hat_k + delta_k

        v_k = B.V[k]
        w_k = B.W[k]

        # Interference
        # interference = sigma2 * np.eye(Nr, dtype=complex)
        # for n in range(K):
        #     if n != k:
        #         H_hat_n = constants.H_HAT[n]
        #         delta_n = A.delta[n].reshape(Nr, Nt)
        #         H_n = H_hat_n + delta_n
        #         v_n = B.V[n]
        #         interference += H_n @ (v_n @ v_n.conj().T) @ H_n.conj().T

        # SINR_k = np.real(w_k.conj().T @ H_k @ v_k + v_k.conj().T @ H_k.conj().T @ w_k \
        #         - w_k.conj().T @ interference @ w_k).item()

        # SINR_k = max(SINR_k, -0.99)  # clip

        # g1_k = np.log(1 + SINR_k)
        g1_k = compute_g1_k(A, B, constants, k)

        delta_penalty = (delta_k.reshape(-1,1).conj().T @ constants.B[k] @ delta_k.reshape(-1,1)).real.item() - 1
        g1_k += B.LAMB[k] * delta_penalty

        total_g1 += g1_k

    # Initialize t
    B.t = 0.5 * total_g1  # for example
    print(f"Initialized t to {B.t:.6e} to ensure feasibility.")

    return B