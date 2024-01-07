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
grid += [i for i in np.arange(0.901, 1.001, 0.001)]
grid = np.array(grid, dtype=np.float64)

ds = np.array([[(grid[i] - grid[i - 1]) * (grid[j] - grid[j - 1]) for i in range(1, len(grid))] for j in
               range(1, len(grid))], dtype=np.float64)
dx = np.sum(ds, axis=1)

grid = grid[:-1]

weights = ds.ravel()

# Grid of values for R2,q
R2_list, q_list = np.meshgrid(grid, grid) # q_list[0]=[0.001, 279 fois]. Ry_list[0]=discretization_grid.
R2_list, q_list = R2_list.ravel(), q_list.ravel() # R2_list=[[0.001,...,0.999], 279 fois] . q_list= [0.001 279 fois, 0.002 279 fois, ... , 0.999 279 fois]
weights = np.outer(weights, weights).ravel() # np.outer = produit pointé, puis sous forme de liste


block1_logfactor_R2 = 0.5 * (-np.log(R2_list) + np.log(1-R2_list))
block1_logfactor_q = np.log(q_list)*1.5 - np.log(1-q_list)

logweights = np.log(weights)
