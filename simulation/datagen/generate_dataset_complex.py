import pickle as pkl
import numpy as np
import argparse
import json
from system import System
from tqdm import tqdm
import os
from joblib import Parallel, delayed


parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='charged',
                    help='What simulation to generate.')
parser.add_argument('--path', type=str, default='data',
                    help='Path to save.')
parser.add_argument('--num_train', type=int, default=2000,
                    help='Number of training simulations to generate.')
parser.add_argument('--num_valid', type=int, default=2000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num_test', type=int, default=2000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=5000,
                    help='Length of trajectory.')
parser.add_argument('--length_test', type=int, default=5000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n_complex', type=int, default=5,
                    help='Number of complex in the simulation.')
parser.add_argument('--average_complex_size', type=int, default=3,
                    help='The expected size of the complex bodies in a system.')
parser.add_argument('--system_types', type=int, default=5,
                    help="The total number of system types.")
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--suffix', type=str, default="",
                    help='add a suffix to the name')
parser.add_argument('--n_workers', type=int, default=1,
                    help="Number of workers")
parser.add_argument('--box_size', type=float, default=None,
                    help="The size of the box.")
parser.add_argument("--config_by_file", default=False, action="store_true", )

args = parser.parse_args()

if args.config_by_file:
    job_param_path = './job_param.json'
    with open(job_param_path, 'r') as f:
        hyper_params = json.load(f)
        args.num_train = hyper_params["num_train"]
        args.num_valid = hyper_params["num_valid"]
        args.num_test = hyper_params["num_test"]
        args.path = hyper_params["path"]
        args.seed = hyper_params["seed"]
        args.n_complex = hyper_params["n_complex"]
        args.average_complex_size = hyper_params["average_complex_size"]
        args.system_types = hyper_params["system_types"]
        args.n_workers = hyper_params["n_workers"]

suffix = '_charged'

suffix += str(args.n_complex) + '_' + str(args.average_complex_size) + '_' + str(args.system_types) + args.suffix
np.random.seed(args.seed)

print(suffix)


def para_comp(length, sample_freq, all_sizes):
    while True:
        X, V = [], []
        system_tyes = len(all_sizes)
        chosen = np.random.randint(0, system_tyes)
        sizes = all_sizes[chosen]
        assert len(sizes) == args.n_complex
        system = System(n_isolated=0, n_stick=0, n_hinge=0,
                        n_complex=args.n_complex, complex_sizes=list(sizes), box_size=args.box_size)

        for t in range(length):
            system.simulate_one_step()
            if t % sample_freq == 0:
                X.append(system.X.copy())
                V.append(system.V.copy())
                # X[t // sample_freq] = system.X.copy()
                # V[t // sample_freq] = system.V.copy()
        system.check()
        # if system.is_valid()  # currently do not apply constraint
        if system.is_valid():
            cfg = system.configuration()
            X = np.array(X)
            V = np.array(V)
            return cfg, X, V, system.edges, system.charges
        else:
            print('Velocity too large, retry')


def generate_dataset(num_sims, length, sample_freq, all_sizes):
    results = Parallel(n_jobs=args.n_workers)(delayed(para_comp)(length, sample_freq, all_sizes) for i in tqdm(range(num_sims)))
    cfg_all, loc_all, vel_all, edges_all, charges_all = zip(*results)
    # print(f'total trials: {cnt:d}, samples: {len(loc_all):d}', cnt)

    return loc_all, vel_all, edges_all, charges_all, cfg_all


if __name__ == "__main__":
    if not os.path.exists(args.path):
        os.mkdir(args.path)

    system_types = args.system_types
    all_sizes = []
    # rand_sizes = np.random.randint(1, args.average_complex_size * 2, args.n_complex)
    targets = np.arange(2, args.average_complex_size * 2 - 1)
    weights = targets[::-1] + 4
    try:
        weights[0] = weights[1]
        weights[2] = weights[1]
    except:
        pass
    weights = weights / weights.sum()
    for _ in range(system_types):
        rand_sizes = np.random.choice(targets, size=args.n_complex, replace=False, p=weights)
        all_sizes.append(rand_sizes)

    print('All sizes:')
    print(all_sizes)

    # exit(0)
    # rand_sizes = [2, 3, 4]
    # rand_sizes = [2, 2, 3, 3]
    # rand_sizes = [3, 4, 5, 6, 7]


    print("Generating {} training simulations".format(args.num_train))
    loc_train, vel_train, edges_train, charges_train, cfg_train = generate_dataset(args.num_train,
                                                                                   args.length,
                                                                                   args.sample_freq,
                                                                                   all_sizes)
    # np.save(os.path.join(args.path, 'loc_train' + suffix + '.npy'), loc_train)
    # np.save(os.path.join(args.path, 'vel_train' + suffix + '.npy'), vel_train)
    # np.save(os.path.join(args.path, 'edges_train' + suffix + '.npy'), edges_train)
    # np.save(os.path.join(args.path, 'charges_train' + suffix + '.npy'), charges_train)
    with open(os.path.join(args.path, 'loc_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(loc_train, f)
    with open(os.path.join(args.path, 'vel_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(vel_train, f)
    with open(os.path.join(args.path, 'edges_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(edges_train, f)
    with open(os.path.join(args.path, 'charges_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(charges_train, f)
    with open(os.path.join(args.path, 'cfg_train' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_train, f)


    print("Generating {} validation simulations".format(args.num_valid))
    loc_valid, vel_valid, edges_valid, charges_valid, cfg_valid = generate_dataset(args.num_valid,
                                                                                   args.length,
                                                                                   args.sample_freq,
                                                                                   all_sizes)
    # np.save(os.path.join(args.path, 'loc_valid' + suffix + '.npy'), loc_valid)
    # np.save(os.path.join(args.path, 'vel_valid' + suffix + '.npy'), vel_valid)
    # np.save(os.path.join(args.path, 'edges_valid' + suffix + '.npy'), edges_valid)
    # np.save(os.path.join(args.path, 'charges_valid' + suffix + '.npy'), charges_valid)
    with open(os.path.join(args.path, 'loc_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(loc_valid, f)
    with open(os.path.join(args.path, 'vel_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(vel_valid, f)
    with open(os.path.join(args.path, 'edges_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(edges_valid, f)
    with open(os.path.join(args.path, 'charges_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(charges_valid, f)
    with open(os.path.join(args.path, 'cfg_valid' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_valid, f)

    print("Generating {} test simulations".format(args.num_test))
    loc_test, vel_test, edges_test, charges_test, cfg_test = generate_dataset(args.num_test,
                                                                              args.length_test,
                                                                              args.sample_freq,
                                                                              all_sizes)
    # np.save(os.path.join(args.path, 'loc_test' + suffix + '.npy'), loc_test)
    # np.save(os.path.join(args.path, 'vel_test' + suffix + '.npy'), vel_test)
    # np.save(os.path.join(args.path, 'edges_test' + suffix + '.npy'), edges_test)
    # np.save(os.path.join(args.path, 'charges_test' + suffix + '.npy'), charges_test)
    with open(os.path.join(args.path, 'loc_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(loc_test, f)
    with open(os.path.join(args.path, 'vel_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(vel_test, f)
    with open(os.path.join(args.path, 'edges_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(edges_test, f)
    with open(os.path.join(args.path, 'charges_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(charges_test, f)
    with open(os.path.join(args.path, 'cfg_test' + suffix + '.pkl'), 'wb') as f:
        pkl.dump(cfg_test, f)
    print('Finished!')

# python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 5 --average_complex_size 3  --n_workers 50
# python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 10 --average_complex_size 5  --n_workers 50
# python -u generate_dataset_complex.py --num-train 2000 --seed 43 --n_complex 15 --average_complex_size 8  --n_workers 50