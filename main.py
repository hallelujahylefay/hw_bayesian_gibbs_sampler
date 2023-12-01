import pickle
from gibbs import gibbs_per_block
from simulate_data import generate_dataset, initialize_parameters

k = 100
s_list = [5, 10, 100]
Ry_list = [0.02, 0.25, 0.5]
no_datasets = 1
datasetsX, datasets = generate_dataset(s_list, Ry_list, no_datasets)

BURNIN_period = 10
ITERATION = 10

res = dict()
for i in datasets.keys():
    X = datasetsX[i]
    dataset = datasets[i]
    for (s, Ry) in dataset.keys():
        Y, beta_v, z_v, sigma2_v, q_v = dataset[(s, Ry)]
        # init = z_v, beta_v, sigma2_v, q_v
        init = initialize_parameters(X, Y)
        res_gibbs = gibbs_per_block(X, Y, init, ITERATION=ITERATION, BURNIN_period=BURNIN_period)
        with open(f'out_{i}_{s}_{Ry}.pickle', 'wb') as handle:
            pickle.dump(res_gibbs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res[i, s, Ry] = res_gibbs

        # whateveryouwant

with open('out.pickle', 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
