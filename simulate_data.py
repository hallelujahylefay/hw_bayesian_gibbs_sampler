import numpy as np

l = 0
k = 100
T = 200
rho = 0.75
no_datasets = 100


def generate_dataset():
    toeplitz_covariance_matrix = np.array([[rho ** np.abs(i - j) for i in range(k)] for j in range(k)])
    X = np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))
    return X
