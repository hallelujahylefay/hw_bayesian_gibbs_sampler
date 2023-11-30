# -*- coding: utf-8 -*-

from scipy.integrate import quad
import numpy as np
import random
from scipy.stats import zscore
from simulate_data import X_data, beta_data, epsilon_data, Y_data, sigma2_data
from tqdm import tqdm
import cond_posteriors as cp

# from simulate_data import generate_dataset
k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1

s_list = [5, 10, 100]
Ry_list = [0.02, 0.25, 0.5]
no_datasets = 100
s = 5
Ry = 0.02
X = X_data()
beta_v = beta_data(s)
epsilon = epsilon_data(Ry, beta_v, X)
Y = Y_data(X, beta_v, epsilon)
sigma2_v = sigma2_data(Ry, beta_v, X)
z_v = (beta_v != 0)
X = zscore(X)
beta_v = zscore(beta_v)
epsilon = zscore(epsilon)
Y = zscore(Y)
(R2_v,q_v)=cp.R2q(X, z_v, beta_v, sigma2_v)()
z_v=cp.z(Y,X,R2_v,q_v)(z_v)
print(z_v)

R2_q = []
for n in tqdm(range(100)):
    
    (R2_v,q_v)=cp.R2q(X, z_v, beta_v, sigma2_v)()
    z_v=cp.z(Y,X,R2_v,q_v)(z_v)
    sigma2_v=cp.sigma2(Y,X,R2_v,q_v,z_v)
    beta_v=cp.betatilde(Y, X, R2_v, q_v, sigma2_v, z_v)
    
    
    
    
