import numpy as np
from scipy.stats import invgamma
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


def gamma2(R2_v, q_v, vbarX_v):
    return R2_v / (q_v * k * vbarX_v * (1 - R2_v))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z_v):
    return X[:, z_v.astype(bool)]


def betahat(Wtilde_v, Xtilde_v, Y):
    return np.linalg.solve(Wtilde_v, Xtilde_v.T @ Y)


def R2q(X, z, beta_v, sigma2_v):
    sz_v = sz(z)
    bz = beta_v @ np.diag(z) @ beta_v.T
    vbarX_v = vbar(X)

    def joint_pdf(q_v, R2_v):
        return np.exp((-bz / (2 * sigma2_v * gamma2(R2_v, q_v, vbarX_v)))) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    @np.vectorize
    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior
        _univariate_pdf = lambda R2_v: joint_pdf(q_v, R2_v)
        return np.sum(_univariate_pdf(grid) * surface)

    def cdf(pdf):
        weights = pdf(grid) * surface
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]
        return cdf

    def invCDF(cdf, u):
        a = np.where(cdf < u)[0]
        if len(a) == 0:
            return grid[-1]
        return grid[a[-1]]

    cdfq = cdf(univariate_pdf)

    def sampleqR():
        u = np.random.uniform(0, 1)
        q_ = invCDF(cdfq, u)

        cdfRconditiononq = cdf(lambda R: joint_pdf(q_, R))
        v = np.random.uniform(0, 1)
        R_ = invCDF(cdfRconditiononq, v)
        return R_, q_

    return sampleqR()  # function that will be looped over to generate samples of (q, R) given X z


def z(Y, X, R2_v, q_v, z_v):
    gamma2_v = gamma2(R2_v, q_v, vbar(X))

    def pdf_ratio(index, z_v):
        """
        Computhe the logratio up to proportionaly of P(Z_i = 1, Z_{-i}) / P(Z_i = 0, Z_{-i})
        """
        z_v[index] = 0
        sz_v = sz(z_v)

        Xtilde_v0 = Xtilde(X, z_v)
        Wtilde_v0 = Wtilde(Xtilde_v0, sz_v, gamma2_v)
        betahat_v0 = betahat(Wtilde_v0, Xtilde_v0, Y)
        _, logdet0 = np.linalg.slogdet(Wtilde_v0)

        z_v[index] = 1
        Xtilde_v1 = Xtilde(X, z_v)
        Wtilde_v1 = Wtilde(Xtilde_v1, sz_v, gamma2_v)
        betahat_v1 = betahat(Wtilde_v1, Xtilde_v1, Y)
        _, logdet1 = np.linalg.slogdet(Wtilde_v1)

        Ysquared = Y.T @ Y

        log_ratio = np.log(q_v) - np.log(1 - q_v) - 1 / 2 * np.log(gamma2_v) - 1 / 2 * (logdet1 - logdet0) - \
                    T / 2(np.log(Ysquared - betahat_v1.T @ Wtilde_v1 @ betahat_v1) - np.log(
            Ysquared - betahat_v0.T @ Wtilde_v0 @ betahat_v0))
        ratio = np.exp(log_ratio)
        return ratio

    def pdf_exclusion(index, z):
        """
        P(z_i | z_{-i}) = 1 / (1+P(z_i | z_{-i})/P(1-z_i | z_{-i}))
        """
        ratio = pdf_ratio(z)
        zi = z[index]
        if zi == 0:
            return - np.log1p(ratio)
        return - np.log1p(1 / ratio)

    def gibbs(z):
        u = np.random.uniform(0, 1, size=k)
        for i in range(k):
            p = pdf_exclusion(i, z)
            if u[i] > p:
                z[i] = 1 - z[i]
        return z

    return gibbs(z_v)


def sigma2(Y, X, R2_v, q_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, vbar(X))
    Xtilde_v = Xtilde(X, z)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    betahat_v = betahat(Wtilde_v, Xtilde_v, Y)
    form = T / 2
    param = (Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) / 2
    # Lorsqu'on regroupera, toute cette initialisation de variables _v ne sera évidemment à faire qu'une fois.
    return invgamma(a=form, scale=param).rvs()  # Ytilde=Y


def betatilde(Y, X, R2_v, q_v, sigma2_v, z_v):
    sz_v = sz(z_v)
    gamma2_v = gamma2(R2_v, q_v, vbar(X))
    Xtilde_v = Xtilde(X, z_v)
    id = np.eye(sz_v)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    Wtilde_v_inv = np.linalg.inv(Wtilde_v)
    mean = Wtilde_v_inv @ Xtilde_v.T @ Y  # Pas de U*phi
    cov = Wtilde_v_inv * sigma2_v
    sample = mnormal(mean, cov).rvs() if sz_v > 0 else np.array([])
    beta_v = np.zeros(shape=k)
    beta_v[z_v] = sample
    return beta_v
