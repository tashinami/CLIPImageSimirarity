import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import warnings
warnings.simplefilter('ignore')

import os
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from CLIP.clip import clip


def arg_parse():
    '''
      各種パラメータの読み込み
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_pretrained_model', default="RN50x4", type=str)
    parser.add_argument('--image_dir_path', default="images", type=str)
    parser.add_argument('--out_dir', default="./result", type=str)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = arg_parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)


    print("CLIPの読み込み...")
    model_clip, preprocess_clip = clip.load(args.clip_pretrained_model, jit=True)  
    model_clip = model_clip.eval()


    print("画像の読み込み...")
    images = []
    image_dir = sorted(glob.glob(args.image_dir_path))
    for image_path in image_dir:
        images.append(preprocess_clip(Image.open(image_path).convert("RGB")))

    image_input = torch.tensor(np.stack(images)).to(device)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]


    print("類似度計算...")
    with torch.no_grad():
      image_features = model_clip.encode_image(image_input).float()

    image_num = len(image_features)
    results = np.ones((image_num, image_num))

    with tqdm(total=image_num, unit="query") as pbar:
      for query in range(image_num):
        for target in range(query, image_num):
          sim = F.cosine_similarity(image_features[query], image_features[target], dim=0)
          results[query][target] = sim.data
          results[target][query] = sim.data


    print("ヒートマップ出力...")
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure()
    sns.heatmap(results, annot=True, square=True, vmax=1.0, vmin=0.0, cmap='Blues')

    result_image_path = os.path.join(args.out_dir, "heatmaps.png")
    plt.savefig(result_image_path)

