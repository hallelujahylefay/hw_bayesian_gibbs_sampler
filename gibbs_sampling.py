import numpy as np
from scipy.stats import invgamma as ig_distrib
from sklearn.linear_model import Lasso
import pickle
import random

# from tqdm.notebook import tqdm

### Function called once

k = 126


def compute_vx(X: np.ndarray) -> float:
    """Compute the average variance of the columns of X."""
    return np.mean(np.var(X, axis=0))


def generate_grid() -> (np.ndarray, np.ndarray, np.ndarray):
    """Generate a discretization grid of values for R2 and q."""
    # Discretization grid
    discretization_grid = [i for i in np.arange(0.001, 0.101, 0.001)]
    discretization_grid += [i for i in np.arange(0.11, 0.91, 0.01)]
    discretization_grid += [i for i in np.arange(0.901, 1, 0.001)]
    discretization_grid = np.array(discretization_grid)
    weights = 0.1 * np.ones(len(discretization_grid))
    weights[(discretization_grid < 0.1) & (discretization_grid > 0.9)] = 0.01

    # Grid of values for R2,q
    R2_list, q_list = np.meshgrid(discretization_grid, discretization_grid)
    R2_list, q_list = R2_list.ravel(), q_list.ravel()
    weights = np.outer(weights, weights).ravel()
    return R2_list, q_list, weights


# Block 1 related

k = 126
R2_list, q_list, weights = generate_grid()
logweights = np.log(weights)


def compute_gamma2_inv_list(vx: float) -> np.ndarray:
    return vx * k * q_list * (1 - R2_list) / R2_list


block1_factor_q = np.power(q_list, 1.5) / (1 - q_list)
block1_factor_R2 = np.power((1 - R2_list) / R2_list, 0.5)
block1_logfactor_R2_q = np.log(block1_factor_q) + np.log(block1_factor_R2)
block1_logfactor_q_2 = np.log(1 - q_list)

### Block 1, called every iteration


def compute_R2_q_probas(
    beta: np.ndarray, z: np.ndarray, sigma2: float, gamma2_inv_list: np.ndarray
) -> np.ndarray:
    """Compute the probabilities of each pair of R2 and q."""
    Is = np.diag(z)
    s = np.sum(z)
    beta2 = float(beta.T @ Is @ beta)
    logprobas = -0.5 / sigma2 * gamma2_inv_list * beta2
    logprobas += s * block1_logfactor_R2_q
    logprobas += k * block1_logfactor_q_2
    logprobas += logweights
    logprobas -= np.max(logprobas)
    probas = np.exp(logprobas)
    probas /= np.sum(probas)
    return probas


def sample_block1(
    beta: np.ndarray,
    z: np.ndarray,
    sigma2: float,
    gamma2_inv_list: np.ndarray,
) -> (np.ndarray, np.ndarray):
    """Sample R2,q as a part of the block 1 of the Gibbs sampler."""
    probas = compute_R2_q_probas(beta, z, sigma2, gamma2_inv_list)
    idx = np.random.choice(range(len(probas)), p=probas)
    return R2_list[idx], q_list[idx]


def compute_gamma2(R2: float, q: float, vx: float) -> float:
    """Compute gamma2 from R2 and q."""
    return R2 / (k * q * vx * (1 - R2))

### Block 2, called every iteration


def compute_Ytilde(Y: np.ndarray, U: np.ndarray, phi :float):
    return Y-U*phi

def sample_phi(Y: np.ndarray, U: np.ndarray, X: np.ndarray, beta : np.ndarray, sigma2: float):
    gram_U = U.T@U
    right = Y-X@beta 
    mean = float((1/gram_U)*U.T@right)
    cov = sigma2*(1/gram_U)
    return np.random.normal(mean,cov)

### Block 3, called every iteration


def compute_W_tilde(X: np.ndarray, z: np.ndarray, gamma2: float) -> np.ndarray:
    """Compute W from X,z and gamma2."""
    X_tilde = X[:, z.astype(bool)]
    W_tilde = X_tilde.T @ X_tilde + np.eye(int(sum(z))) / gamma2
    return W_tilde


def compute_beta_tilde_hat(
    W_tilde: np.ndarray, X_tilde: np.ndarray, Y_tilde: np.ndarray
) -> np.ndarray:
    beta_tilde_hat = np.linalg.inv(W_tilde) @ X_tilde.T @ Y_tilde
    return beta_tilde_hat



def compute_logproba_z(
    z: np.ndarray, X: np.ndarray, Y: np.ndarray, U : np.ndarray, phi: float, gamma2: float, q, T=376
) -> float:
    """Compute the logprobability of z_i given the other parameters, unormalized."""
    s = np.sum(z)
    logproba = s * (np.log(q) - np.log(1 - q))
    logproba += -s / 2 * np.log(gamma2)
    W_tilde = compute_W_tilde(X, z, gamma2)
    logproba += -1 / 2 * np.log(np.linalg.det(W_tilde))
    X_tilde = X[:, z.astype(bool)]
    Y_tilde = compute_Ytilde(Y, U, phi)
    beta_tilde_hat = compute_beta_tilde_hat(W_tilde, X_tilde, Y_tilde)
    Ysquared = Y_tilde.T @ Y_tilde
    _ = Ysquared - beta_tilde_hat.T @ W_tilde @ beta_tilde_hat
    logproba += (-T / 2) * np.log(_ / 2)      
    return logproba


def sample_z_i(
    z: np.ndarray, X: np.ndarray, Y: np.ndarray, U : np.ndarray, phi: float, gamma2: float, q: float, idx: int
):
    """Sample z_i given the other parameters."""
    # Compute unnormalized logprobas
    z[idx] = 0
    logproba_z_i_0 = compute_logproba_z(z, X, Y, U, phi, gamma2, q)
    z[idx] = 1
    logproba_z_i_1 = compute_logproba_z(z, X, Y, U, phi, gamma2, q)

    # Sample
    logprobas = np.array([logproba_z_i_0, logproba_z_i_1])
    logprobas -= np.max(logprobas)
    probas = np.exp(logprobas)
    probas /= np.sum(probas)

    u = np.random.uniform()
    if u < probas[0]:
        z[idx] = 0
    else:
        z[idx] = 1


def sample_block3(
    z: np.ndarray, X: np.ndarray, Y: np.ndarray, U : np.ndarray, phi: float, gamma2: float, q: float, n_iter=1
):
    """Sample z via Gibbs sampling,
    as a part of the block 3 of the Gibbs sampler."""
    for iter in range(n_iter):
        for idx in range(len(z)):
            sample_z_i(z, X, Y, U, phi, gamma2, q, idx)


def compute_R2(q: float, k: int, gamma2: float, vx: float) -> float:
    _ = q * k * gamma2 * vx
    return _ / (1 + _)


### Block 4, called every iteration


def sample_block4(
    Y: np.ndarray, U : np.ndarray, W_tilde: np.ndarray, beta_tilde_hat: np.ndarray, phi: float, sigma2: float, T=376
) -> float:
    """Sample sigma2 given the other parameters.
    https://en.wikipedia.org/wiki/Inverse-gamma_distribution"""
    # a (shape)
    a = T / 2
    # b (scale)
    Y_tilde = compute_Ytilde(Y,U,phi)
    _ = beta_tilde_hat.T @ W_tilde @ beta_tilde_hat
    b = 0.5 * float(Y_tilde.T @ Y_tilde - _)
    assert b > 0.0, b
    sigma2 = ig_distrib.rvs(a, scale=b)
    return sigma2


### Block 5, called every iteration


def sample_block5(
    beta_tilde_hat: np.ndarray, W_tilde: np.ndarray, sigma2: float, z: np.ndarray
) -> np.ndarray:
    """Sample beta given the other parameters."""
    # mean
    mean = beta_tilde_hat.flatten()
    # cov
    cov = sigma2 * np.linalg.inv(W_tilde)

    # print("beta_tilde_hat:", beta_tilde_hat)
    # print("W_tilde det:", np.linalg.det(W_tilde))

    # sample
    beta_tilde = np.random.multivariate_normal(mean, cov)
    beta = np.zeros(len(z))
    beta[z.astype(bool)] = beta_tilde
    return beta


### Gibbs sampler









def gibbs_sampler(X: np.ndarray, Y: np.ndarray, U: np.ndarray, init,  n_iter: int, burn: int):
    # Initialize
    vx = compute_vx(X)
    gamma2_inv_list = compute_gamma2_inv_list(vx)
    z, beta, phi, sigma2, q = init
    params = []
    number_inclusions=np.zeros(len(z))

    for iter in range(n_iter):
        # Block 1
        R2, q = sample_block1(beta, z, sigma2, gamma2_inv_list)
        gamma2 = compute_gamma2(R2, q, vx)
        
        
        # Block 2
        phi = sample_phi(Y, U, X, beta, sigma2)
        Y_tilde = compute_Ytilde(Y, U, phi)
        
        # Block 3
        sample_block3(z, X, Y, U, phi, gamma2, q, n_iter=1)

        if np.sum(z) == 0:
            W_tilde = np.zeros((1, 1))
            beta_tilde_hat = np.zeros((1, 1))

            # Block 4
            sigma2 = sample_block4(Y, W_tilde, beta_tilde_hat, sigma2)

            # Block 5
            beta = np.zeros((X.shape[1], 1))

        else:
            W_tilde = compute_W_tilde(X, z, gamma2)
            beta_tilde_hat = compute_beta_tilde_hat(W_tilde, X[:, z.astype(bool)], Y_tilde)

            # Block 4
            sigma2 = sample_block4(Y,U, W_tilde, beta_tilde_hat,phi, sigma2)

            # Block 5
            beta = sample_block5(beta_tilde_hat, W_tilde, sigma2, z)
        number_inclusions += z
        if iter > burn:
            params.append([R2, q, phi, gamma2, sigma2])
            

    return params , number_inclusions
