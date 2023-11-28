import numpy as np
import random
from scipy.stats import zscore
import pandas as pd

k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1
s_list = [5, 10, 100]
Ry_list = [0.2, 0.25, 0.5]
no_datasets = 100
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



def X_data(rho=0.75):
    toeplitz_covariance_matrix = np.array([[rho ** abs(i - j) for i in range(k)] for j in range(k)])
    np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))
    return np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))

def beta_data(s):
    beta = np.random.normal(0, 1, size=k)
    zeroes_position = random.sample(range(0, 100),
                                    k - s)  # s nombres, choisis au hasard entre 0 et k. les coordonnées correspondantes dans beta seront rendues nulles.
    beta[zeroes_position] = 0
    return beta
    

def epsilon_data(Ry, beta, X):
    s_bxt = np.sum((beta.T @ X) ** 2, axis=0) / T
    return np.random.normal(0, (1 / Ry - 1) * (1 / T) * s_bxt, size=T)


def Y_data(X, beta, epsilon):
    Y = X @ beta + epsilon
    return Y

#
# def generate_dataset(s_list=[5, 10, 100], Ry_list=[0.2, 0.25, 0.5], no_datasets=100):
#     """
#     Retourne les datasets correspondants à toutes les valeurs possibles du couple (s,Ry) sous forme de dictionnaire. Pour obtenir le dataset correspondant à s=a et Ry=b, il faut écrire datasets[a,b].
#
#
#     """
#
#     # Triple liste: on itère sur les valeurs de s et de Ry pour créer un dataset
#     datasets = {}
#     for s in s_list:
#         for Ry in Ry_list:
#             dataset = pd.DataFrame(index=["Dataset " + str(i + 1) for i in range(no_datasets)],
#                                    columns=["X", "beta", "epsilon", "Y"])
#
#             for i in range(no_datasets):
#                 X = X_data()
#                 beta = beta_data(s)
#                 epsilon = epsilon_data(Ry, beta, X)
#                 Y = Y_data(X, beta, epsilon)
#
#                 X = zscore(X)
#                 beta = zscore(beta)
#                 epsilon = zscore(epsilon)
#                 Y = zscore(Y)
#                 # we standardize the data, column by column before putting it in our dataset.
#                 dataset.loc["Dataset " + str(i + 1)] = [X, beta, epsilon, Y]
#             datasets[(s, Ry)] = dataset
#
#     return datasets
#
#
# datasets = generate_dataset()

"""
    
    #Triple liste: on itère sur les valeurs de s et de Ry pour créer un dataset
    datasets={}
    for s in s_list:
        for Ry in Ry_list: 
            dataset=pd.DataFrame(index=["Dataset " +str(i+1) for i in range(no_datasets)],columns=["X","beta","epsilon","Y","z"])

            for i in range(no_datasets):
                X=X_data()
                beta=beta_data(s)
                epsilon=epsilon_data(Ry,beta,X)
                Y=Y_data(X,beta,epsilon)
                z=beta==0
                #z est un array de booléens, automatiquement remplacé par les valeurs correspondantes lorsque c'est nécessaire.
                #z est déterminé dès maintenant, puisqu'il est plus compliqué de le déterminer une fois beta normalisé. 
                X=zscore(X,axis=0)# Un doute sur l'axe ici: est-ce qu'on normalise pour T ou pour K? 
                beta=zscore(beta)
                epsilon=zscore(epsilon)
                Y=zscore(Y)
                #we standardize the data, column by column before putting it in our dataset.

                dataset.loc["Dataset " +str(i+1)]=[X,beta,epsilon,Y,z]
            datasets[(s,Ry)]=dataset.to_numpy()
               


datasets=generate_dataset()
"""