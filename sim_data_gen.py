# import numpy as np

# Nt=32
# Nr=2
# K=2


# H_HAT = np.array([
#     np.array([
#             np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt)) +
#             1j * np.random.normal(0, 1 / np.sqrt(2), (Nr, Nt))
#             for _ in range(K)
#         ])
#  for _ in range(300)])


# dim = Nr * Nt

# Delta_k = np.array([[np.random.normal(0, 1/np.sqrt(2), (Nr, Nt)) + 1j * np.random.normal(0, 1/np.sqrt(2), (Nr, Nt)) for _ in range(K)]for _ in range(10000)])

# # mult by 10**(snrestdb/10) to obtain mismatch following corresponding snrest

# from scipy.io import savemat

# savemat('H_HAT.mat', {'H_HAT': H_HAT})

# savemat('Delta_k.mat', {'Delta_k': Delta_k})
import numpy as np
from scipy.io import loadmat

data = loadmat('rates.mat')

print("Robust mean rates (r_m):", data['r_m'].flatten())
print("Robust rate variances (r_v):", data['r_v'].flatten())
print("Non-robust mean rates (n_m):", data['n_m'].flatten())
print("Non-robust rate variances (n_v):", data['n_v'].flatten())
print("WMMSE mean rates (w_m):", data['w_m'].flatten())
print("WMMSE rate variances (w_v):", data['w_v'].flatten())