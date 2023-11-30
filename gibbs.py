# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from simulate_data import generate_dataset
from tqdm import tqdm
import cond_posteriors as cp

k = 100

s_list = [5]
Ry_list = [0.02]
no_datasets = 1
datasetsX, datasets = generate_dataset(s_list, Ry_list, no_datasets)

BURNIN_period = 100
ITERATION = 200
R2_vs = np.zeros((ITERATION,))
q_vs = np.zeros((ITERATION,))
z_vs = np.zeros((ITERATION, k))
sigma2_vs = np.zeros((ITERATION,))
for i in datasets.keys():
    X = datasetsX[i]
    dataset = datasets[i]
    for (s, Ry) in dataset.keys():
        Y, beta_v, z_v, sigma2_v = dataset[(s, Ry)]
        print(sigma2_v)
        for n in tqdm(range(ITERATION + BURNIN_period)):
            R2_v, q_v = cp.R2q(X, z_v, beta_v, sigma2_v)()
            z_v = cp.z(Y, X, R2_v, q_v)(z_v)
            sigma2_v = cp.sigma2(Y, X, R2_v, q_v, z_v)
            beta_v_tilde = cp.betatilde(Y, X, R2_v, q_v, 1.0, z_v)
            if n >= BURNIN_period:
                R2_vs[n - BURNIN_period] = R2_v
                q_vs[n - BURNIN_period] = q_v
                z_vs[n - BURNIN_period] = z_v
                sigma2_vs[n - BURNIN_period] = sigma2_v

print(len(R2_vs))
plt.hist(R2_vs, bins=10)
plt.savefig('R2.png')
plt.close()

plt.hist(q_vs, bins=10)
plt.savefig('q.png')
plt.close()

plt.hist(sigma2_vs, bins=10)
plt.savefig('sigma2.png')
plt.close()
