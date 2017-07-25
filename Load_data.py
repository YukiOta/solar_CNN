# coding: utf-8
"""solar_projectにおける、データ整理用プログラム
1.発電量csvファイルから、必要な列を取り出して、整理する
2.画像の読み込み
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image


def load_image(imgdir, size, norm=True):
    """
    input: image directory
    output: numpy array
    """
    images = []
    for filename in os.listdir(imgdir):
        img = Image.open(os.path.join(imgdir, filename))
        if img is not None:
            img = img.resize(size)
            img = np.array(img)
            if norm is True:
                img = img / 255.
            images.append(img)
    img_array = np.array(images)
    print("Image Load Done")
    return img_array


def load_target(csv):
    """
    csvファイルを読み込む
    パンダスを用いて、処理する
    """
    df = pd.read_csv(csv, header=None)
    target = []
    target.append(df.values[:, 0])
    target.append(df.values[:, 1])
    target = np.array(target, np.float64)
    print("CSV Load Done")
    return target


df = pd.read_csv("/Users/yukiota/solar_project/data/PV_CSV/201611/Trd20161117_1.csv")
df = df.ix[:, [0, 11, 36]]
df.columns = ["time", "power", "temperature"]
df[0:2]




































# end
