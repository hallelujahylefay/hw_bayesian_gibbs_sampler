import numpy as np

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
grid = np.array(grid)
weights = 0.1 * np.ones(len(grid))
weights[(grid < 0.1) | (grid > 0.9)] = 0.01

# Grid of values for R2,q
R2_list, q_list = np.meshgrid(grid, grid)
R2_list, q_list = R2_list.ravel(), q_list.ravel()
weights = np.outer(weights, weights).ravel()
block1_factor_q = np.power(q_list, 1.5) / (1 - q_list)
block1_factor_R2 = np.power((1 - R2_list) / R2_list, 0.5)
block1_logfactor_R2 = np.log(block1_factor_q) + np.log(block1_factor_R2)
block1_logfactor_q = np.log(1 - q_list)
logweights = np.log(weights)
