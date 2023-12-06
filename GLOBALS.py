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
grid = np.array(grid, dtype=np.float64)

ds = np.array([[(grid[i] - grid[i - 1]) * (grid[j] - grid[j - 1]) for i in range(1, len(grid))] for j in
               range(1, len(grid))], dtype=np.float64)
dx = np.sum(ds, axis=1)
grid = grid[:-1]

