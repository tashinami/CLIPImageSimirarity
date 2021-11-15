
import os
import argparse
import numpy as np

from matplotlib import pyplot as plt
from IPython.display import display

import cv2
from PIL import Image
from google.colab.patches import cv2_imshow


def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--query_image_idx', default=0, type=int)
    parser.add_argument('--image_dir_path', default="./images", type=str)
    parser.add_argument('--out_dir', default="./result", type=str)
    parser.add_argument('--idx_width', default=5, type=int)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()

    results = np.load(os.path.join(args.out_dir, 'results.npy'))

    simirarity_list = results[args.query_image_idx]
    simirarity_idx_list = np.argsort(-simirarity_list)

    score_high_idx = simirarity_idx_list[:args.idx_width]
    score_low_idx = simirarity_idx_list[-args.idx_width:]

    # 類似度の高い画像のリスト
    for idx in score_high_idx:
      image_path = "image_" + str(idx+1).zfill(5) + ".jpg"
      print(simirarity_list[idx], image_path)

      image = plt.imread(os.path.join(args.image_dir_path, image_path))
      display(image)


      # image = cv2.imread(os.path.join(args.image_dir_path, image_path))
      # cv2_imshow(image)
      # cv2.waitKey(0)

    # 類似度の低い画像のリスト
    for idx in score_low_idx:
      image_path = "image_" + str(idx+1).zfill(5) + ".jpg"
      print(simirarity_list[idx], image_path)
      image = cv2.imread(os.path.join(args.image_dir_path, image_path))
      plt.imshow(image)



