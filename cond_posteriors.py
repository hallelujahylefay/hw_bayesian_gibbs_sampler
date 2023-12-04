import numpy as np
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mnormal

k = 100
T = 200
l = 0.
a = 1.
b = 1.
A = 1.
B = 1.

grid = [i for i in np.arange(0.001, 0.101, 0.001)]
grid += [i for i in np.arange(0.11, 0.91, 0.01)]
grid += [i for i in np.arange(0.901, 1, 0.001)]
grid = np.array(grid, dtype=np.float64)
surface = np.zeros(len(grid), dtype=np.float64)
surface[:-1] = np.sum(
    np.array([[(grid[i] - grid[i - 1]) * (grid[j] - grid[j - 1]) for i in range(1, len(grid))] for j in
              range(1, len(grid))], dtype=np.float64), axis=1)
surface[-1] = surface[-2]


def sz(z):
    return np.sum(z)


def vbar(X):
    return np.mean(np.var(X, axis=0))


def gamma2(R2_v, q_v, X):
    return R2_v / (q_v * k * vbar(X) * (1 - R2_v))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z_v):
    return X[:, z_v.astype(bool)]


def betahat(Wtilde_v, Xtilde_v, Y):
    WtildeinvXtilde_v = np.linalg.solve(Wtilde_v, Xtilde_v.T)
    betahat_v = WtildeinvXtilde_v @ Y
    return betahat_v


def R2q(X, z, beta_v, sigma2_v):
    sz_v = sz(z)
    bz = beta_v @ np.diag(z) @ beta_v.T
    # bz = np.einsum('i, i ->', beta_v ** 2, z)
    vbarX_v = vbar(X)

    def joint_pdf(q_v, R2_v):
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbarX_v * q_v * ((1 - R2_v) / R2_v) * bz)) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    @np.vectorize
    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior
        _univariate_pdf = lambda R2_v: joint_pdf(q_v, R2_v)
        return np.sum(_univariate_pdf(grid) * surface)

    def cdf(pdf):
        weights = pdf(grid) * surface
        normalize_constant = np.sum(weights)
        weights /= normalize_constant
        cdf = np.cumsum(weights)
        return cdf

    def invCDF(cdf, u):
        return grid[np.where(cdf < u)[0][-1]]

    cdfq = cdf(univariate_pdf)

    def sampleqR():
        u = np.random.uniform(0, 1)
        q_ = invCDF(cdfq, u)

        cdfRconditiononq = cdf(lambda R: joint_pdf(q_, R))
        v = np.random.uniform(0, 1)
        R_ = invCDF(cdfRconditiononq, v)
        return q_, R_

    return sampleqR  # function that will be looped over to generate samples of (q, R) given X z


def z(Y, X, R2_v, q_v):
    gamma2_v = gamma2(R2_v, q_v, X)

    def logpdf(z_v):
        sz_v = sz(z_v)
        Xtilde_v = Xtilde(X, z_v)
        Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
        betahat_v = betahat(Wtilde_v, Xtilde_v, Y)
        _, logdet = np.linalg.slogdet(Wtilde_v)
        logp = sz_v * (np.log(q_v) - np.log(1 - q_v)) - sz_v / 2 * np.log(gamma2_v) - 1 / 2 * logdet \
               - T / 2 * np.log((Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) / 2)
        return logp

    def logpdf_exclusion(index, z):
        logp = logpdf(z)
        zi = z[index]
        if zi == 0:
            logp0 = logp
            z[index] = 1
            logp1 = logpdf(z)
        else:
            logp1 = logp
            z[index] = 0
            logp0 = logpdf(z)
        z[index] = zi
        return logp - np.logaddexp(logp0, logp1)
        # return logp - np.logaddexp(logp0 + np.log(q_v), logp1 + np.log(1 - q_v))

    def pdf_exclusion(i, z):
        return np.exp(logpdf_exclusion(i, z))

    def gibbs(z):
        u = np.random.uniform(0, 1, size=k)
        for i in range(k):
            p = pdf_exclusion(i, z)
            if u[i] > p:
                z[i] = 1 - z[i]
        return z

    return gibbs


def sigma2(Y, X, R2_v, q_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    betahat_v = betahat(Wtilde_v, Xtilde_v, Y)
    form = T / 2
    param = (Y.T @ Y - betahat_v.T @ (Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v) @ betahat_v) / 2
    scale = 1 / param
    # Lorsqu'on regroupera, toute cette initialisation de variables _v ne sera évidemment à faire qu'une fois.
    return 1 / (gamma(a=form).rvs() * scale)  # Ytilde=Y


def betatilde(Y, X, R2_v, q_v, sigma2_v, z_v):
    sz_v = sz(z_v)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z_v)
    id = np.eye(sz_v)
    invTerm = np.linalg.inv(id / gamma2_v + Xtilde_v.T @ Xtilde_v)
    mean = invTerm @ Xtilde_v.T @ Y  # Pas de U*phi
    cov = invTerm * sigma2_v
    sample = mnormal(mean, cov).rvs()
    beta_v = np.zeros(shape=k)
    beta_v[z_v] = sample
    return beta_v
