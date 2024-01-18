import random
from sklearn.linear_model import Lasso
import numpy as np

k = 100
T = 200


def X_data(rho=0.75):
    indices = np.arange(k)
    toeplitz_covariance_matrix = rho ** np.abs(indices[:, None] - indices)
    return np.random.normal(size=(T, k)) @ np.linalg.cholesky(toeplitz_covariance_matrix)


def beta_data(s):
    beta = np.random.normal(0, 1, size=k)
    zeroes_position = random.sample(range(k),
                                    k - s)
    beta[zeroes_position] = 0
    return beta


def sigma2_data(Ry, beta, X):
    return (1 / Ry - 1) * np.mean((X @ beta) ** 2, axis=0)


def generate_dataset(s_list, Ry_list, no_datasets):
    """
    Retourne les datasets correspondants à toutes les valeurs possibles du couple (s,Ry) sous forme de dictionnaire. Pour obtenir le dataset correspondant à s=a et Ry=b, il faut écrire datasets[a,b].


    """

    # Triple liste: on itère sur les valeurs de s et de Ry pour créer un dataset
    datasets = dict()
    for i in range(no_datasets):
        datasets[i] = dict()
        X = X_data()
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)  # standardize the data
        for s in s_list:
            for Ry in Ry_list:
                beta = beta_data(s)
                sigma2 = sigma2_data(Ry, beta, X)
                epsilon = np.random.normal(0, sigma2 ** 0.5, size=T)
                z = (beta != 0)
                q = np.sum(z) / k
                Y = X @ beta + epsilon
                datasets[i][(s, Ry)] = X, Y, beta, z, sigma2, q

    return datasets


def initialize_parameters(X, Y):
    lasso_reg = Lasso(alpha=0.1, fit_intercept=False)
    lasso_reg.fit(X, Y)
    beta = lasso_reg.coef_
    y_pred = lasso_reg.predict(X)
    residuals = Y - y_pred
    sigma2 = np.var(residuals)
    z = np.where(beta != 0, True, False)
    q = np.sum(z) / len(z)
    # gamma2 = np.var(beta[z.astype(bool)]) / sigma2
    return z, beta, sigma2, q
