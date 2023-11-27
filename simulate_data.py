import numpy as np
import pandas as pd 
import random 

k = 100
T = 200
l = 0
a = 1
b = 1
A = 1
B = 1
s_list=[5,10,100]
Ry_list=[0.2,0.25,0.5]
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
def X_data(rho=0.75):
    toeplitz_covariance_matrix = np.array([[rho ** abs(i - j) for i in range(k)] for j in range(k)])
    np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))
    return np.linalg.cholesky(toeplitz_covariance_matrix) @ np.random.normal(size=(k, T))

def beta_data(s):
    beta=np.random.normal(0,1,size=k)
    zeroes_position=random.sample(range(0,100),k-s) # s nombres, choisis au hasard entre 0 et k. les coordonnées correspondantes dans beta seront rendues nulles.
    beta[zeroes_position]=0
    return beta
    

def epsilon_data(Ry,beta,X):
    X_t = np.array([X[:,i] for i in range(T)])
    temp=[beta.T @ X_t[i] for i in range(T)]
    betaxt=list(map(lambda x:x**2,temp))
    return np.random.normal(0, (1/Ry-1) * (1/T)* sum(betaxt),size=T)    

    
def Y_data(X,beta,epsilon):
    X_t = np.array([X[:,i] for i in range(T)])
    return np.array([X_t[i].T @ beta +epsilon[i] for i in range(T)])

def generate_dataset(s_list=[5,10,100],Ry_list=[0.2,0.25,0.5],no_datasets=100):
    """
    Retourne les datasets correspondants à toutes les valeurs possibles du couple (s,Ry) sous forme de dictionnaire. Pour obtenir le dataset correspondant à s=a et Ry=b, il faut écrire datasets[a,b].
    

    """
    
    #Triple liste: on itère sur les valeurs de s et de Ry pour créer un dataset
    datasets={}
    for s in s_list:
        for Ry in Ry_list: 
            dataset=pd.DataFrame(index=["Dataset " +str(i+1) for i in range(no_datasets)],columns=["X","beta","epsilon","Y"])

            for i in range(no_datasets):
                X=X_data()
                beta=beta_data(s)
                epsilon=epsilon_data(Ry,beta,X)
                Y=Y_data(X,beta,epsilon)
                dataset.loc["Dataset " +str(i+1)]=[X,beta,epsilon,Y]
            datasets[(s,Ry)]=dataset     

            
    return datasets

datasets=generate_dataset()