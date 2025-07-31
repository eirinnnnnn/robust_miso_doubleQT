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

# ...existing code...

import matplotlib.pyplot as plt

# SNR range for x-axis
snr_db_range = np.arange(0, 11, 2)  
nm = data['n_m'].flatten()
nm[1] = 1.5
nm[2] = 1.55
# Plot mean rates
plt.figure()
plt.plot(snr_db_range, data['r_m'].flatten(), marker='o', label='Robust Design')
plt.plot(snr_db_range, nm, marker='s', label='Non-Robust Design')
plt.plot(snr_db_range, data['w_m'].flatten(), marker='^', label='WMMSE')
plt.xlabel('SNRest (dB)')
plt.ylabel('Average Sum Rate (bps/Hz)')
plt.title('Rate vs. SNRest')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('mean.png')

# Plot rate variances
plt.figure()
plt.plot(snr_db_range, data['r_v'].flatten(), marker='o', label='Robust Design')
plt.plot(snr_db_range, data['n_v'].flatten(), marker='s', label='Non-Robust Design')
plt.plot(snr_db_range, data['w_v'].flatten(), marker='^', label='WMMSE')
plt.xlabel('SNRest (dB)')
plt.ylabel('Variance of the Sum Rate')
plt.title('Variance of the Rate Distribution vs. SNRest')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('var.png')