import numpy as np
import cond_posteriors as cp

k = 126
frequency = 100


def gibbs_per_block(X, Y, U, init, ITERATION=200, BURNIN_period=100):
    R2_vs = np.zeros((ITERATION,))
    q_vs = np.zeros((ITERATION,))
    z_vs = np.zeros((ITERATION, k))
    sigma2_vs = np.zeros((ITERATION,))
    beta_vs = np.zeros((ITERATION, k))
    phi_vs = np.zeros((ITERATION,))

    z_v, beta_v, phi_v, sigma2_v, q_v = init
    for n in range(ITERATION + BURNIN_period):
        if (n + 1) % frequency == 0:
            print(f"Iter: {n + 1}")
        R2_v, q_v = cp.R2q(X, z_v, beta_v, sigma2_v)
        phi_v=cp.phi( Y, U, X, beta_v, sigma2_v)
        z_v = cp.z(Y, U, X, phi_v, R2_v, q_v, z_v)
        sigma2_v = cp.sigma2(Y, U, X, phi_v, R2_v, q_v, z_v)
        beta_v = cp.betatilde(Y, U, X, phi_v, R2_v, q_v, sigma2_v, z_v)
        if n >= BURNIN_period:
            R2_vs[n - BURNIN_period] = R2_v
            q_vs[n - BURNIN_period] = q_v
            phi_vs[n - BURNIN_period] = phi_v
            z_vs[n - BURNIN_period] = z_v
            sigma2_vs[n - BURNIN_period] = sigma2_v
            beta_vs[n - BURNIN_period] = beta_v
    return R2_vs, q_vs, phi_vs, z_vs, sigma2_vs, beta_v

