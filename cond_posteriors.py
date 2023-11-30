import numpy as np
from scipy.stats import gamma
from scipy.stats import multivariate_normal as mnormal

grid_q = np.array(
    [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in range(900, 1000)])
grid_R2 = np.array([i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in
                                                                                           range(900, 1000)])

k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1


def sz(z):
    return np.sum(z)


def vbar(X):
    return np.mean(np.var(X, axis=0))


def gamma2(R2_v, q_v, X):
    return (1 / (1 - R2_v) - 1) * 1 / (q_v * k * vbar(X))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z_v):
    return X[:, z_v == 1]


def sigma2_data(beta_v, X, Ry):  # le sigma_2 qui serviraà générer le jeu de données.
    return (1 / (Ry - 1)) * np.mean((beta_v @ X.T) ** 2, axis=0)


def betahat(Wtildeinv_v, Xtilde_v, Y):
    return Wtildeinv_v @ Xtilde_v.T @ Y


def R2q(X, z, beta_v, sigma2_v):
    sz_v = sz(z)
    bz = beta_v @ np.diag(z) @ beta_v.T
    vbarX_v = vbar(X)

    def joint_pdf(q_v, R2_v):
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbarX_v * q_v * ((1 - R2_v) / R2_v) * bz)) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    @np.vectorize
    def univariate_pdf(q_v):
        # marginal of q, integrate joint posterior
        _univariate_pdf = lambda R2_v: joint_pdf(q_v, R2_v)
        return np.sum(_univariate_pdf(grid_R2))

    def conditional_pdf(q_v, R2_v):
        # distribution of q conditional on R2, proportionate to the joint posterior
        return joint_pdf(q_v, R2_v) / univariate_pdf(q_v)

    def cdf(pdf, grid):
        weights = pdf(grid)
        normalize_constant = np.sum(weights)
        weights /= normalize_constant
        cdf = np.cumsum(weights)
        return cdf

    def invCDF(cdf, grid, u):
        return grid[np.where(cdf < u)[0][-1]]

    cdfq = cdf(univariate_pdf, grid_q)

    def sampleqR():
        u = np.random.uniform(0, 1)
        q_ = invCDF(cdfq, grid_q, u)

        cdfQconditiononR = cdf(lambda R: conditional_pdf(q_, R), grid_R2)
        v = np.random.uniform(0, 1)
        R_ = invCDF(cdfQconditiononR, grid_R2, v)
        return q_, R_

    return sampleqR  # function that will be looped over to generate samples of (q, R) given X z


def z(Y, X, R2_v, q_v):
    def logpdf(z_v):
        sz_v = sz(z_v)
        gamma2_v = gamma2(R2_v, q_v, X)
        Xtilde_v = Xtilde(X, z_v)
        Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
        Wtildeinv_v = np.linalg.inv(Wtilde_v)
        betahat_v = betahat(Wtildeinv_v, Xtilde_v, Y)
        logp = sz_v * np.log(q_v) + (k - sz_v) * np.log(1 - q_v) - sz_v / 2 * np.log(gamma2_v) - 1 / 2 * np.log(
            np.linalg.det(Wtilde_v)) \
               - T / 2 * np.log((Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) / 2)
        return logp

    def logpdf_exclusion(index, z):
        logp = logpdf(z)
        z[index] = 0
        logp0 = logpdf(z)
        z[index] = 1
        logp1 = logpdf(z)
        return logp - np.logaddexp(logp0 + np.log(q_v), logp1 + np.log(1 - q_v))

    def iter_gibbs(z):
        for i in range(k):
            logp = logpdf_exclusion(i, z)
            p = np.exp(logp)
            u = np.random.uniform(0, 1)
            if u > p:
                z[i] = 0
            else:
                z[i] = 1
        return z

    def gibbs(init, iter=1):
        z = init
        for i in range(iter):
            z = iter_gibbs(z)
        return z

    return gibbs


def sigma2(Y, X, R2_v, q_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    Wtildeinv_v = np.linalg.inv(Wtilde_v)
    betahat_v = betahat(Wtildeinv_v, Xtilde_v, Y)
    form = T / 2
    param = (Y.T @ Y - betahat_v.T @ (Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v )@ betahat_v) / 2
    scale = 1 / param
    # Lorsqu'on regroupera, toute cette initialisation de variables _v ne sera évidemment à faire qu'une fois.
    return 1/(gamma(a=form).rvs()*scale)  # Ytilde=Y


def betatilde(Y, X, R2_v, q_v, sigma2_v, z):
    sz_v = sz(z)
    gamma2_v = gamma2(R2_v, q_v, X)
    Xtilde_v = Xtilde(X, z)
    id = np.eye(sz_v)
    invTerm = np.linalg.inv(id / gamma2_v + Xtilde_v.T @ Xtilde_v)
    mean = invTerm @ Xtilde_v.T @ Y  # Pas de U*phi
    cov = invTerm * sigma2_v
    return mnormal(mean, cov).rvs()
