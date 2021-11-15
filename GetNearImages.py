
import os
import cv2
import argparse
import numpy as np

from matplotlib import pyplot as plt

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

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
    image_list = []
    for idx in score_high_idx:
      image_path = "image_" + str(idx+1).zfill(5) + ".jpg"
      image = cv2.imread(os.path.join(args.image_dir_path, image_path))

      # 画像名とスコアの描画
      string = "image_" + str(idx+1).zfill(5) + " score:" + str(round(simirarity_list[idx], 2))
      image = cv2.rectangle(image, (0, 0), (300, 30), (0, 0, 0), -1)
      image = cv2.putText(image, string, (5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

      image_list.append(image)

    socre_high_image = hconcat_resize_min(image_list)


    # 類似度の低い画像のリスト
    image_list = []
    for idx in score_low_idx:
      image_path = "image_" + str(idx+1).zfill(5) + ".jpg"
      image = cv2.imread(os.path.join(args.image_dir_path, image_path))

      # 画像名とスコアの描画
      string = "image_" + str(idx+1).zfill(5) + " score:" + str(round(simirarity_list[idx], 2))
      image = cv2.rectangle(image, (0, 0), (300, 30), (0, 0, 0), -1)
      image = cv2.putText(image, string, (5, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=(255, 255, 255), thickness=2)

      image_list.append(image)

    socre_low_image = hconcat_resize_min(image_list)


    result_image = vconcat_resize_min([socre_high_image, socre_low_image])
    # cv2.imwrite(os.path.join(args.out_dir, 'matching_result.png'), result_image)
    cv2.imshow("result", result_image)
    cv2.waitKey(-1)



