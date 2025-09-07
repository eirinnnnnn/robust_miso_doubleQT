import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the .mat file
# data = loadmat('rates_snrest_snr5.mat')
data = loadmat('rates.mat')

# Extract arrays and convert to numpy arrays for easy slicing
r_m = np.array(data['r_m'])  # shape: (num_snr, K)
r_v = np.array(data['r_v'])
# n_m = np.array(data['n_m'])
# n_v = np.array(data['n_v'])
# w_m = np.array(data['w_m'])
# w_v = np.array(data['w_v'])
r_o = np.array(data['r_o'])
# n_o = np.array(data['n_o'])
# w_o = np.array(data['w_o'])

# snr_db_range = np.arange(-10, 20, 5)  # adjust if needed
snr_db_range = np.array(data['snr'][0]) 
K = r_m.shape[1]
user_labels = [f'User {k+1}' for k in range(K)]

designs = [
    ('Robust', r_m, r_v, r_o, 'tab:blue', 'o'),
    # ('Non-Robust', n_m, n_v, n_o, 'tab:orange', 's'),
    # ('WMMSE', w_m, w_v, w_o, 'tab:green', '^'),
]

# --- Plot Mean Rate ---
plt.figure(figsize=(8, 5))
for name, mean, _, _, color, marker in designs:
    # for k in range(K):
    #     plt.plot(snr_db_range, mean[:, k], label=f"{name} ({user_labels[k]})", color=color, marker=marker, linestyle='-' if k==0 else '--')
    plt.plot(snr_db_range, np.mean(mean, axis=1), label=f"{name}", color=color, marker=marker)
plt.xlabel("SNR (dB)")
plt.ylabel("Mean Rate (bps/Hz)")
plt.title("Mean Rate vs. SNR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean.png")

# --- Plot Rate Variance ---
plt.figure(figsize=(8, 5))
for name, _, var, _, color, marker in designs:
    # for k in range(K):
    #     plt.plot(snr_db_range, var[:, k], label=f"{name} ({user_labels[k]})", color=color, marker=marker, linestyle='-' if k==0 else '--')
    plt.plot(snr_db_range, np.mean(var, axis=1), label=f"{name}", color=color, marker=marker)
plt.xlabel("SNR (dB)")
plt.ylabel("Rate Variance")
plt.title("Rate Variance vs. SNR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("var.png")

# --- Plot Outage Rate ---
plt.figure(figsize=(8, 5))
for name, _, _, outage, color, marker in designs:
    # for k in range(K):
    #     plt.plot(snr_db_range, outage[:, k], label=f"{name} ({user_labels[k]})", color=color, marker=marker, linestyle='-' if k==0 else '--')
    plt.plot(snr_db_range, np.mean(outage, axis=1), label=f"{name}", color=color, marker=marker)
plt.xlabel("SNR (dB)")
plt.ylabel("Outage Rate (bps/Hz)")
plt.title("Outage Rate vs. SNR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outage.png")