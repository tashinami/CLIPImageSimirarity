
import argparse
import numpy as np


def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image_idx', default=0, type=int)
    parser.add_argument('--image_dir_path', default="./images", type=str)
    parser.add_argument('--out_dir', default="./result", type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()

    results = np.load("results.npy")
    print(results)

    sim_list = results[args.query_image_idx]
    for idx, simirarity in enumerate(sim_list):
      print(idx, simirarity)