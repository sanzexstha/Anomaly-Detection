import itertools
import os
import argparse
import random
import pandas as pd
from torch.backends import cudnn
from utils.utils import *

from solver import Solver
import wandb


def str2bool(v):
    return v.lower() in ('true')


def main(config):
    # cudnn.benchmark = True
    if (not os.path.exists(config.model_save_path)):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))

    # Hyperparameter candidates
    # local_window_list = [5, 7, 9, 11]
    # stride_list = [10, 15, 20, 30]
    # local_window_list = [3, 5, 7]
    # stride_list = [9, 11, 13]
    #
    # results = []
    # # Grid Search
    # for local_window, stride in itertools.product(local_window_list, stride_list):
    #   print(f"\n Trying local_window={local_window}, stride={stride}")
    #   config.local_window = local_window
    #   config.stride = stride
    #   solver.train()
    #   _, _, _, f1_score = solver.test()
    #   # Save the result
    #   results.append({
    #     "local_window": local_window,
    #     "stride": stride,
    #     "f1_score": float(f"{f1_score:.4f}"),
    #   })
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("dozer_grid_search_results.csv", index=False)
    #
    # print("\n🔎 All Results:")
    # print(results_df)
    #
    # # Find best
    # best = results_df.sort_values(by="f1_score", ascending=False).iloc[0]
    # print(f"\n🏆 Best setting found: local_window={best.local_window}, stride={best.stride}, F1={best.f1_score:.4f}")
    if config.mode == 'train':
        solver.train()
        torch.cuda.empty_cache()

    elif config.mode == 'test':
        solver.test()
        torch.cuda.empty_cache()

    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2023, help='Random Seed')
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--n_vars', type=int, default=55)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='credit')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/creditcard_ts.csv')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--anormly_ratio', type=float, default=4.00)
    parser.add_argument("--d_model", type=int, default=512)


    # DozerAttention parameters
    parser.add_argument('--local_window', type=int, default=0, help='The size of local window')
    parser.add_argument('--stride', type=int, default=17, help='The stride interval sparse attention. If set to 24, interval will be 24.')
    parser.add_argument('--rand_rate', type=int, default=0.1, help='The rate of random attention')
    parser.add_argument('--vary_len', type=int, default=1, help='The start varying length, if 1 input equals output')
    # parser.add_argument('--sparse', type=bool, default=False, help='Introduce sparsity')

    # parser.add_argument('--Msl_run', type=int, default='1', help='Introduce sparsity')


    parser.add_argument('--wandb', type=bool, default=False, help='flag for whether use wandb')

    # parser.add_argument('--', type=bool, default=True, help='flag for whether use wandb')


    # Data embedding
    parser.add_argument("--data_embed", type=str, default="local")
    parser.add_argument("--k_size", type=int, default=3)  # local embedding


    config = parser.parse_args()
    # fix the seed for reproducibility, default 2023
    seed = config.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if config.wandb==True:
        wandb.login()
        wandb.init(project="DozerAnomaly", config=config)

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    main(config)
