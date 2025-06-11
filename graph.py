import numpy as np
import matplotlib.pyplot as plt

# Data
snr_est_range = np.arange(0, 20, 5)
robust = [5.01976907574878, 5.05119981293425, 4.895645691651476, 4.8263411628523265]
nonrobust = [4.024616978126102, 4.551964105685285, 4.7692519109361005, 4.781606423910016]

# Plot
plt.figure()
plt.plot(snr_est_range, robust, marker='o', label='Robust Design')
plt.plot(snr_est_range, nonrobust, marker='s', label='Non-Robust Design')
plt.xlabel('SNR_est (dB)')
plt.ylabel('Rate (bps/Hz)')
plt.title('Rate vs. SNR_est')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("QAQ.png")
