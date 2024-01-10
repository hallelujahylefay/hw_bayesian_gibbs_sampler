from GLOBALS import *


def sz(z_v):
    return np.sum(z_v)


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
    bz = beta_v.T @ np.diag(z) @ beta_v
    vbarX_v = vbar(X)
    logprobas = - bz / (2 * sigma2_v * gamma2(R2_list, q_list, vbarX_v))
    logprobas += sz_v * block1_logfactor_R2
    logprobas += k * block2_logfactor_R2
    logprobas += logweights
    logprobas -= np.max(logprobas)
    probas = np.exp(logprobas)
    probas /= np.sum(probas)
    idx = np.random.choice(range(len(probas)), p=probas)
    return R2_list[idx], q_list[idx]


def z(Y, X, R2_v, q_v, z_v):
    gamma2_v = gamma2(R2_v, q_v, vbar(X))

    def pdf_ratio(index, z_v):
        """
        Computhe the ratio of P(z_i = 1, z_{-i}) / P(z_i = 0, z_{-i})
        """
        z_v[index] = 0
        sz_v = sz(z_v)

        Xtilde_v0 = Xtilde(X, z_v)
        Wtilde_v0 = Wtilde(Xtilde_v0, sz_v, gamma2_v)
        betahat_v0 = betahat(Wtilde_v0, Xtilde_v0, Y)
        _, logdet0 = np.linalg.slogdet(Wtilde_v0)

        z_v[index] = 1
        sz_v += 1
        Xtilde_v1 = Xtilde(X, z_v)
        Wtilde_v1 = Wtilde(Xtilde_v1, sz_v, gamma2_v)
        betahat_v1 = betahat(Wtilde_v1, Xtilde_v1, Y)
        _, logdet1 = np.linalg.slogdet(Wtilde_v1)

        Ysquared = Y.T @ Y

        log_ratio = np.log(q_v) - np.log1p(- q_v) - 1 / 2 * np.log(gamma2_v) - 1 / 2 * (logdet1 - logdet0) - \
                    T / 2 * (np.log(Ysquared - betahat_v1.T @ Wtilde_v1 @ betahat_v1) - np.log(
            Ysquared - betahat_v0.T @ Wtilde_v0 @ betahat_v0))
        ratio = np.exp(log_ratio)
        return ratio

    def pdf_exclusion(index, z):
        """
        P(z_i | z_{-i}) = 1 / (1+P(1-z_i, z_{-i})/P(z_i, z_{-i}))
        """
        ratio = pdf_ratio(index, z)
        if not z[index]:
            return np.exp(- np.log1p(ratio))
        return np.exp(- np.log1p(1 / ratio))

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
    param = (Y.T @ Y - betahat_v.T @ Wtilde_v @ betahat_v) / 2
    return 1 / np.random.gamma(shape=T / 2, scale=1 / param)


def betatilde(Y, X, R2_v, q_v, sigma2_v, z_v):
    sz_v = sz(z_v)
    gamma2_v = gamma2(R2_v, q_v, vbar(X))
    Xtilde_v = Xtilde(X, z_v)
    Wtilde_v = Wtilde(Xtilde_v, sz_v, gamma2_v)
    Wtilde_v_inv = np.linalg.inv(Wtilde_v)
    mean = Wtilde_v_inv @ Xtilde_v.T @ Y  # Pas de U*phi
    cov = Wtilde_v_inv * sigma2_v
    sample = np.random.multivariate_normal(mean, cov) if sz_v > 0 else np.array([])
    beta_v = np.zeros(shape=k)
    beta_v[z_v] = sample
    return beta_v
