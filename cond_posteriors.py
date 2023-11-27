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
    return np.mean(np.var(X, axis=0))


def gamma2(R2_v, q_v, X):
    return (1 / (1 - R2_v) - 1) * 1 / (q_v * k * vbar(X))


def Wtilde(Xtilde_v, sz_v, gamma2_v):
    return Xtilde_v.T @ Xtilde_v + np.eye(sz_v) / gamma2_v


def Xtilde(X, z):
    return X[:, z == 1]

def sigma2(T_v,Ry_v,X):
    return (1/(Ry-1)) * sum((beta @ X.T)**2)/T

def R2q(X, z, n):
    def posteriorR2q(q_v, R2_v, X, z, beta, sigma2_v):
        bz = beta @ np.diag(z) @ beta.T
        sz_v = sz(z)
        sigma2_v= sigma2(T,Ry,X)
        return np.exp((-1 / (2 * sigma2_v)) * (k * vbar(X) * q_v * ((1 - R2_v) / R2_v) * bz)) * q_v ** (
                3 / 2 * sz_v + a - 1) * (
                       1 - q_v) ** (k - sz_v + b - 1) * R2_v ** (A - 1 - sz_v / 2) * (1 - R2_v) ** (sz_v / 2 + B - 1)

    grid_q = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in range(900, 1000)]
    grid_R2 = [i / 1000 for i in range(1, 100)] + [i / 100 for i in range(10, 90)] + [i / 1000 for i in
                                                                                      range(900, 1000)]
    # initial values for q and R2
    q_ = 0.9
    R_ = 0.1

    q = []
    R = []

    for j in range(n):
        w_r = [posteriorR2q(i, q_, X, z) for i in grid_R2]
        s = np.sum(w_r)
        w_r = [i / s for i in w_r]
        cdf_r = np.cumsum(w_r)
        u = np.random.uniform(0, 1)
        R_ = grid_R2[np.argmax(cdf_r > u)]
        R.append(R_)

        w_q = [posteriorR2q(R_, i, X, z) for i in grid_q]
        s = np.sum(w_q)
        w_q = [i / s for i in w_q]
        cdf_q = np.cumsum(w_q)
        v = np.random.uniform(0, 1)
        q_ = grid_q[np.argmax(cdf_q > v)]

        q.append(q_)
    return list(zip(q, R))


def betahat(Wtildeinv_v, Xtilde_v, Y):
    return Wtildeinv_v @ Xtilde_v.T @ Y


def z(Y, X, R2_v, q_v):
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
        return pdf(z) / (q_v * pdf(z_copy1) + (1 - q_v) * pdf(z_copy0))


def sigma2(Y, X, R2, q, z):
    # rayane
    raise NotImplementedError


def betatilde(Y, X, R2, q, sigma2, z):
    # rayane
    raise NotImplementedError
