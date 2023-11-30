from simulate_data import generate_dataset
from gibbs import gibbs_per_block

k = 100

s_list = [5]
Ry_list = [0.02]
no_datasets = 1
datasetsX, datasets = generate_dataset(s_list, Ry_list, no_datasets)

BURNIN_period = 100
ITERATION = 200

res = dict()
for i in datasets.keys():
    X = datasetsX[i]
    dataset = datasets[i]
    for (s, Ry) in dataset.keys():
        Y, beta_v, z_v, sigma2_v = dataset[(s, Ry)]
        init = (z_v, beta_v, sigma2_v, 0.5)
        res_gibbs = gibbs_per_block(X, Y, init, ITERATION=ITERATION, BURNIN_period=BURNIN_period)
        res[i, s, Ry] = res_gibbs
        #whateveryouwant
