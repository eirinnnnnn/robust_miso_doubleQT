def update_B_loop(A, B, constants, outer_tol=1e-3, max_outer_iter=50, inner_tol=1e-3, max_inner_iter=20):
    """
    Outer loop for updating B (V, Lambda, t, alpha, beta).
    Inside each outer iteration, run a micro-inner loop on (w, v) until micro-lagrangian converges.
    """

    K = constants.K
    Nr = constants.NR
    Pt = constants.PT

    # Initialize auxiliary variables
    w = [np.random.randn(Nr, 1) + 1j*np.random.randn(Nr, 1) for _ in range(K)]

    prev_outer_lagrangian = None
    lagrangian_trajectory = []
    converged = False

    for outer_iter in range(max_outer_iter):
        # === Micro inner loop: update (w, v) alternately ===
        prev_micro_lagrangian = None
        for micro_iter in range(max_inner_iter):
            # Update w
            for k in range(K):
                w[k] = update_w(A, B, constants, k)

            # Update v
            for n in range(K):
                B.V[n] = update_v(A, B, constants, w, n)

            # Compute Lagrangian after w,v update
            current_micro_lagrangian = compute_lagrangian_B(A, B, constants, w)
            print(f"    [Micro iter {micro_iter+1}] Micro-lagrangian = {current_micro_lagrangian:.6e}")

            if prev_micro_lagrangian is not None:
                delta_micro = abs(current_micro_lagrangian - prev_micro_lagrangian)
                print(f"        Micro Lagrangian change = {delta_micro:.6e}")
                if delta_micro < inner_tol:
                    print(f"    ✅ Micro-inner loop converged at iteration {micro_iter+1}.")
                    break

            prev_micro_lagrangian = current_micro_lagrangian

        # === Step 2: Update lagrangian multipliers ===
        B = update_lagrangian_variables(A, B, constants, w)

        # === Step 3: Compute outer lagrangian ===
        current_outer_lagrangian = compute_lagrangian_B(A, B, constants, w)
        lagrangian_trajectory.append(current_outer_lagrangian)

        print(f"[B Outer iter {outer_iter+1}] Outer Lagrangian = {current_outer_lagrangian:.6e}")

        if prev_outer_lagrangian is not None:
            delta_outer = abs(current_outer_lagrangian - prev_outer_lagrangian)
            print(f"    Outer Lagrangian change = {delta_outer:.6e}")
            if delta_outer < outer_tol:
                print(f"✅ Outer loop converged at iteration {outer_iter+1}.")
                converged = True
                break

        prev_outer_lagrangian = current_outer_lagrangian

    if not converged:
        print(f"⚠️ Outer loop reached max iterations without full convergence.")

    return B, lagrangian_trajectory
