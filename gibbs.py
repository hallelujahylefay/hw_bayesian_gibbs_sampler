import numpy as np
import cond_posteriors as cp
from tqdm import tqdm

k = 100


def gibbs_per_block(X, Y, init, ITERATION=200, BURNIN_period=100):
    R2_vs = np.zeros((ITERATION,))
    q_vs = np.zeros((ITERATION,))
    z_vs = np.zeros((ITERATION, k))
    sigma2_vs = np.zeros((ITERATION,))
    beta_tilde_vs = np.zeros((ITERATION, k))

    z_v, beta_v, sigma2_v, q_v = init
    for n in tqdm(range(ITERATION + BURNIN_period)):
        R2_v, q_v = cp.R2q(X, z_v, beta_v, sigma2_v)
        z_v = cp.z(Y, X, R2_v, q_v, z_v)
        sigma2_v = cp.sigma2(Y, X, R2_v, q_v, z_v)
        beta_v = cp.betatilde(Y, X, R2_v, q_v, sigma2_v, z_v)
        if n >= BURNIN_period:
            R2_vs[n - BURNIN_period] = R2_v
            q_vs[n - BURNIN_period] = q_v
            z_vs[n - BURNIN_period] = z_v
            sigma2_vs[n - BURNIN_period] = sigma2_v
            beta_tilde_vs[n - BURNIN_period] = beta_v
    return R2_vs, q_vs, z_vs, sigma2_vs, beta_v
