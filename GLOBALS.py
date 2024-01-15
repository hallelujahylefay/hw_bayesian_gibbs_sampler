import numpy as np

k = 127 
T = 376
l = 1.
a = 1.
b = 1.
A = 1.
B = 1.

grid = np.concatenate((np.arange(0.001, 0.101, 0.001), np.arange(0.11, 0.91, 0.01), np.arange(0.901, 1, 0.001)),
                      dtype=np.float64)
weights = 0.1 * np.ones(len(grid))
weights[(grid < 0.1) | (grid > 0.9)] = 0.01

# Grid of values for R2,q
R2_list, q_list = np.meshgrid(grid, grid)
R2_list, q_list = R2_list.ravel(), q_list.ravel()

weights = np.outer(weights, weights).ravel()
logweights = np.log(weights)

block1_logfactor_R2 = 1.5 * np.log(q_list) - np.log(1 - q_list) + 0.5 * (np.log(1 - R2_list) - np.log(R2_list))
block2_logfactor_R2 = np.log(1 - q_list)
