import numpy as np
import random
from scipy.stats import zscore
from sklearn.linear_model import Lasso

k = 100
T = 200


def X_data(rho=0.75):
    indices = np.arange(k)
    toeplitz_covariance_matrix = rho ** np.abs(indices[:, None] - indices)
    # toeplitz_covariance_matrix = np.array([[rho ** abs(i - j) for i in range(k)] for j in range(k)])
    return np.random.normal(size=(T, k)) @ np.linalg.cholesky(toeplitz_covariance_matrix)


def beta_data(s):
    beta = np.random.normal(0, 1, size=k)
    zeroes_position = random.sample(range(0, 100),
                                    k - s)  # s nombres, choisis au hasard entre 0 et k. les coordonnées correspondantes dans beta seront rendues nulles.
    beta[zeroes_position] = 0
    return beta


def sigma2_data(Ry, beta, X):
    s_bxt = np.sum((X @ beta) ** 2, axis=0) / T
    return (1 / Ry - 1) * (1 / T) * s_bxt


def generate_dataset(s_list, Ry_list, no_datasets):
    """
    Retourne les datasets correspondants à toutes les valeurs possibles du couple (s,Ry) sous forme de dictionnaire. Pour obtenir le dataset correspondant à s=a et Ry=b, il faut écrire datasets[a,b].


    """

    # Triple liste: on itère sur les valeurs de s et de Ry pour créer un dataset
    datasets = dict()
    datasets_X = np.zeros(shape=(no_datasets, T, k))
    for i in range(no_datasets):
        datasets[i] = dict()
        X = X_data()
        datasets_X[i] = X
        for s in s_list:
            for Ry in Ry_list:
                beta = beta_data(s)
                sigma2 = sigma2_data(Ry, beta, X)
                epsilon = np.random.normal(0, sigma2 ** 0.5, size=T)
                Y = X @ beta + epsilon
                z = (beta != 0)
                X = zscore(X)
                beta = zscore(beta)
                Y = zscore(Y)
                # we standardize the data, column by column before putting it in our dataset.
                datasets[i][(s, Ry)] = Y, beta, z, sigma2

    return datasets_X, datasets
