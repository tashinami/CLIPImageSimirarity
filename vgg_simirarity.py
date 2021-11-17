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
from torchvision import models, transforms


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

def extract(model, target, inputs):
    feature = None

    def forward_hook(module, inputs, outputs):
        # 順伝搬の出力を features というグローバル変数に記録する
        global features
        # 1. detach でグラフから切り離す。
        # 2. clone() でテンソルを複製する。モデルのレイヤーで ReLU(inplace=True) のように
        #    inplace で行う層があると、値がその後のレイヤーで書き換えられてまい、
        #    指定した層の出力が取得できない可能性があるため、clone() が必要。
        features = outputs.detach().clone()

    # コールバック関数を登録する。
    handle = target.register_forward_hook(forward_hook)

    # 推論する
    model.eval()
    model(inputs)

    # コールバック関数を解除する。
    handle.remove()

    return features

if __name__ == "__main__":
    args = arg_parse()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("VGGの読み込み...")
    model = models.vgg16(pretrained=True).to(device)
    target_module = model.classifier[3]
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("画像の読み込み...")
    images = []
    image_dir = sorted(glob.glob(os.path.join(args.image_dir_path, "*.jpg")))
    image_dir = image_dir[:10]
    for image_path in image_dir:
      image = Image.open(image_path)
      processed_image = preprocess(image)
      processed_image = processed_image.unsqueeze(0).to(device)
      images.append(processed_image)


    print("類似度計算...")
    image_features = []
    with torch.no_grad():
      for image in images:
        feature = extract(model, target_module, image)
        image_features.append(feature[0])
        

    image_num = len(image_features)
    results = np.ones((image_num, image_num))

    with tqdm(total=image_num, unit="query") as pbar:
      for query in range(image_num):
        for target in range(query, image_num):
          sim = F.cosine_similarity(image_features[query], image_features[target], dim=0)
          results[query][target] = sim.data
          results[target][query] = sim.data

        pbar.update(1)


    print("ヒートマップ出力...")
    os.makedirs(args.out_dir, exist_ok=True)
    plt.figure()
    sns.heatmap(results, annot=True, square=True, vmax=1.0, vmin=0.0, cmap='Blues')

    result_image_path = os.path.join(args.out_dir, "heatmaps.png")
    plt.savefig(result_image_path)

    # ファイル出力
    np.save(os.path.join(args.out_dir, 'results.npy'), results)

