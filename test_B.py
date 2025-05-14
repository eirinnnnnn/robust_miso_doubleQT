import numpy as np
import matplotlib.pyplot as plt
from variables import GlobalConstants, VariablesA, VariablesB, initialize_t
from update_functions import update_A_loop, update_B_loop

# Set up a reproducible experiment with 1 user
constants = GlobalConstants(K=1, Nt=4, Nr=4, snr_db=10, snrest_db=11, Pt=16)
A = VariablesA(constants)
B = VariablesB(constants)
B = initialize_t(A, B, constants)

# For tracking multipliers
alpha_log = []
beta_log = []
lambda_log = []
t_log = []
lagrangian_log = []

# Modified B-loop that logs internal multipliers
outer_tol = 1e-6
max_outer_iter = 10000
inner_tol = 1e-4
max_inner_iter = 500

for outer_iter in range(max_outer_iter):
    A, _ = update_A_loop(A, B, constants, inner_tol=inner_tol, max_inner_iter=max_inner_iter, plot_lagrangian=False)
    B, _ = update_B_loop(A, B, constants, outer_tol=outer_tol, max_outer_iter=1, inner_tol=inner_tol, max_inner_iter=max_inner_iter)

    alpha_log.append(B.ALPHA)
    beta_log.append(B.BETA)
    lambda_log.append(B.LAMB.copy())
    t_log.append(B.t)
    # Note: This uses the compute_lagrangian_B from your file directly
    from update_functions import compute_lagrangian_B
    lagrangian_log.append(compute_lagrangian_B(A, B, constants))

import pandas as pd
df = pd.DataFrame({
    'iteration': np.arange(1, len(alpha_log)+1),
    'alpha': alpha_log,
    'beta': beta_log,
    'lambda_0': [l[0] for l in lambda_log],
    't': t_log,
    'Lagrangian': lagrangian_log
})
import matplotlib.pyplot as plt

# Create a plot for each Lagrangian multiplier and the Lagrangian value
fig, axs = plt.subplots(5, 1, figsize=(10, 18), sharex=True)

iters = df['iteration']

# Plot alpha
axs[0].plot(iters, df['alpha'], marker='o')
axs[0].set_ylabel(r'$\alpha$')
axs[0].grid(True)
axs[0].set_title('Evolution of Lagrangian Multipliers and Lagrangian Value')

# Plot beta
axs[1].plot(iters, df['beta'], marker='o', color='orange')
axs[1].set_ylabel(r'$\beta$')
axs[1].grid(True)

# Plot lambda_0
axs[2].plot(iters, df['lambda_0'], marker='o', color='green')
axs[2].set_ylabel(r'$\lambda_0$')
axs[2].grid(True)

# Plot t
axs[3].plot(iters, df['t'], marker='o', color='purple')
axs[3].set_ylabel(r'$t$')
axs[3].grid(True)

# Plot Lagrangian
axs[4].plot(iters, df['Lagrangian'], marker='o', color='red')
axs[4].set_ylabel(r'$\mathcal{L}_2$')
axs[4].set_xlabel('Iteration')
axs[4].grid(True)

plt.tight_layout()
plt.savefig("lagrangian_multipliers.png")


# from ace_tools import display_dataframe_to_user
# display_dataframe_to_user(name="Lagrangian Multiplier Trace", dataframe=df)
