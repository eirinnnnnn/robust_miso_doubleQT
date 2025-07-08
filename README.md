# Robust MU-MIMO Beamforming Optimization

This project implements a robust beamforming design for multi-user MIMO systems under channel uncertainty, using a primal-dual algorithm based on quadratic transforms (QT) and Lagrangian optimization.

## 📌 Overview

The system jointly optimizes:
- Auxiliary variables \(\mathbf{y}_k, \mathbf{w}_k\)
- Beamforming vectors \(\mathbf{v}_k\)
- Mismatch perturbations \(\delta_k\)
- Lagrangian multipliers \(\alpha, \beta, \lambda_k\)
- Surrogate variable \(t\)

The formulation supports **robust design** (with CSI uncertainty modeled by ellipsoidal sets) and **non-robust design** (assuming perfect CSI).

The optimization is structured in two nested loops:
1. **Algorithm 1 (A-update)**: optimizes mismatch \(\delta_k\) and auxiliary variables.
2. **Algorithm 2 (B-update)**: optimizes beamformers \(\mathbf{v}_k\), dual variables, and objective surrogate.

## 📁 File Structure

- `functions.py`: core algorithmic components (QT formulation, gradient updates, dual ascent, rate computation).
- `variables.py`: model parameter setup and initializations.
- `test_B_KKT.py`: example script to run and compare robust vs. non-robust design.

## ▶️ How to Run

To evaluate both robust and non-robust optimization:

```bash
python3 test_B_KKT.py
