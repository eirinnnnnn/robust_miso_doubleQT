import numpy as np
import scipy.stats as stats


def load_constants(
        snr_db=10, 
        snrest_db=11, 
        Nt=4, Nr=4, K=4,
        Pt=16
    ):
    """
    Load configuration-specific constants.
    Args:
       snr_db: SNR of the AWGN
       snrest_db: SNR of the channel estiment 
    Returns:
        constants: a dictionary of global matrices and scalars
    """

    sigma = (10**(-snr_db/10))**(1/2)
    sigmaest = (10**(-snrest_db/10))**(1/2)

    h_hat = []
    for _ in range(K):
        h_hat.append(np.random.normal(0, 0.5, (Nr, Nt)) + 1j*np.random.normal(0, 0.5, (Nr, Nt)))
    B = np.eye(Nt*Nr)/sigmaest**2
    constants = {
        "NT": Nt,
        "NR": Nr,
        "K": K,
        "PT": Pt,

        ### constant variables
        "SNR_DB": snr_db,
        "SIGMA": sigma,
        "SNREST_DB": snrest_db,
        "SIGMAEST": sigmaest,
        ### constant matrices
        "H_HAT": h_hat,
        "B": B 
    }
    return constants
