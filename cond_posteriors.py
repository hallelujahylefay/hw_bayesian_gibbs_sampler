import numpy as np

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
    raise NotImplementedError


def gamma2(R2_v, q_v, X):
    return (1 / (1 - R2_v) - 1) * 1 / (q_v * k * vbar(X))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z):
    return X[:, z == 1]


def R2q(Y, U, X, theta, z):
    # hamza
    raise NotImplementedError


def betahat(Wtildeinv_v, Xtilde_v, Y):
    return Wtildeinv_v @ Xtilde_v.T @ Y


def z(Y, U, X, R2_v, q_v):
    def pdf(z):
        sz_v = sz(z)
        gamma2_v = gamma2(R2_v, q_v, X)
        Xtilde_v = Xtilde(X, z)
        Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
        Wtildeinv_v = np.linalg.inv(Wtilde_v)
        betahat_v = betahat(Wtildeinv_v, Xtilde_v, Y)
        q_v ** sz_v * (1 - q_v) ** (k - sz_v) * (1 / gamma2_v) ** (sz_v / 2) * np.linalg.det(Wtildeinv_v) ** (1 / 2) \
        * (Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) ** (-T / 2)

    def pdf_exclusion(index, z):
        # not optimal
        z_copy0 = z.copy()
        z_copy1 = z.copy()
        z_copy0[index] = 0
        z_copy1[index] = 1
        return pdf(z) / (q_v * pdf(z_copy1) + (1-q_v) * pdf(z_copy0))


def sigma2(Y, U, X, R2, q, z):
    # rayane
    raise NotImplementedError


def betatilde(Y, U, X, R2, q, sigma2, z):
    # rayane
    raise NotImplementedError
