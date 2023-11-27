import numpy as np


k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1
rho=0.75
no_datasets=100
"""
J'imagine la syntaxe comme suit:
s=5,10 ou 100
Ry=2%,5% ou 50%
X = ce que la prof demande

beta_data=beta_data(s)
epsilon_data=epsilon_data(Ry)
On tire ce dataset 100 fois, on fait ce qu'on doit faire.

On recommence avec
s ou Ry qui change de valeur (ou les deux)
Et ainsi de suite jusqu'à avoir couvert toutes les valeurs possibles du couple (s,Ry)

"""    

toeplitz_covariance_matrix = np.array([[rho ** abs(i - j) for i in range(k)] for j in range(k)])
X = np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))
X_t = np.array([X[i] for i in range(k)])

def beta_data(s):
    beta=np.random.normal(0,1,size=k)
    zeroes_position=np.random.randint(0,k,size=s) # s nombres, choisis au hasard entre 0 et k. les coordonnées correspondantes dans beta seront rendues nulles.
    beta[zeroes_position]=0
    return beta
    

def epsilon(Ry,beta):
    return np.random.normal(0,(Ry-1)^-1*sum(1/T*(beta.T@X_t[i])^2 for i in range()),size=T)    

    
def Y_data(X,beta_data,epsilon):
    return np.array([X_t[i].T @ beta_data +epsilon[i] for i in range(T)])