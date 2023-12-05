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

surface = np.zeros(len(grid), dtype=np.float64)
surface[:-1] = np.sum(
    np.array([[(grid[i] - grid[i - 1]) * (grid[j] - grid[j - 1]) for i in range(1, len(grid))] for j in
              range(1, len(grid))], dtype=np.float64), axis=1)
surface[-1] = surface[-2]
