from simulate_data import generate_dataset
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:05:40 2023

@author: rayan
"""

"Imaginons que la grosse boucle est faite et tout."

s_list = [5, 10, 100]
Ry_list = [0.2, 0.25, 0.5]
no_datasets = 100
datasets = generate_dataset(s_list, Ry_list, no_datasets)

for dataset_s_Ry in datasets.values():
    # On a 100 datasets avec des valeurs fixées de s et Ry
    for dataset in dataset_s_Ry.index:
        X_eps_Y, beta_z = dataset_s_Ry.loc[dataset]
        q_med = []
        """
        On lance notre grosse boucle 110 000 fois, et on retire les 10 000 premières étapes.
        On garde toutes les autres étapes dans des vecteurs contenant tous nos paramètres
        On récupère la médiane de q, on la met dans q_med
        On fait ça pour chaque dataset, et on plot le résultat sous forme d'histogramme.
        Au-dessus de l'histogramme, on récupère aussi la distribution théorique de la médiane
        """

    q_distrib = 0
    plt.hist(q_med)
    plt.plot(q_distrib)
    plt.show()  # On voudra peut-être des subplots, à voir

    """
    Question 2:
    On récupère un dataset au hasard, pour lequel on va plot la posterior marginal distribution de q.
    J'avoue ne pas avoir compris ce qu'elle veut dire par "plot the marginal of q by using the histogram.
    """
    rand = np.random.randint(1, 101)
    X_rand, beta_rand, epsilon_rand, Y_rand = dataset_s_Ry.loc[f"Dataset {rand}"]
    # De là, on peut faire ce qu'elle a demandé. Je sais pas trop comment parce que j'ai pas compris, mais on peut.

    """
    Question 3: 
        A aviser quand on aura les résultats.
    """
