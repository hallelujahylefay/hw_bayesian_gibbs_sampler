import pickle
from gibbs import gibbs_per_block
from simulate_data import generate_dataset, initialize_parameters
from multiprocessing import Process

s_list = [100]
Ry_list = [0.5]
no_datasets = 1

output_path = "./OK"
multiprocess = False

BURNIN_period = 1000
ITERATION = 1000


def run(dataset, identifier, output_path="."):
    for (s, Ry) in dataset.keys():
        X, Y, beta_v, z_v, sigma2_v, q_v = dataset[(s, Ry)]
        init = initialize_parameters(X, Y)
        print(f"(i, s, Ry): {identifier, s, Ry}, sigma²: {init[2]}, q: {init[3]}")
        res_gibbs = X, Y, gibbs_per_block(X, Y, init, ITERATION=ITERATION, BURNIN_period=BURNIN_period)
        with open(f'{output_path}/{identifier}_{s}_{Ry}.pickle', 'wb') as handle:
            pickle.dump(res_gibbs, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    datasets = generate_dataset(s_list, Ry_list, no_datasets)
    if multiprocess:
        procs = []
        for i in datasets.keys():
            procs.append(Process(target=run, args=(datasets[i], i, output_path)))
        for proc in procs:
            proc.start()
    else:
        for i in datasets.keys():
            run(datasets[i], i, output_path)
