# coding: utf-8
"""ZCA白色化のプログラム
試しに、1日分のデータを用いて白色化してみる (6月16日かな)
画像の正規化は、全画像の平均を引いて255で割ってスケーリング
"""

import numpy as np
import argparse
import time
from PIL import Image
import matplotlib.pyplot as plt
import Load_data as ld
from scipy import ndimage, linalg


def compute_mean(image_array):
    """全画像の平均をとって、平均を返す
    入力：画像データセット (np配列を想定してる) [枚数, height, width, rgb]
    出力：平均画像
    """
    print("conmpute mean image")
    mean_image = np.ndarray.mean(image_array, axis=0)
    return mean_image

# class ZCA_Whitening:
#     def __init__(self, epsilon=1E-6):
#         self.epsilon = epsilon
#         self.mean = None


def main():
    print("start image load")
    st_im = time.time()
    test_img_dir = "../data/PV_IMAGE/201706/20170616_IT_resampled_northup"
    img_0616 = ld.load_image(test_img_dir, (100, 100))
    elp_im_time = time.time() - st_im
    print("elapsed_time:{0}".format(elp_im_time)+" [sec]")

    mean_img = compute_mean(img_0616)
    tmp = img_0616 - mean_img

    tmp = tmp.reshape(tmp.shape[0], -1)
    mean = np.mean(tmp)
    tmp -= mean
    cov_mat = np.dot(tmp.T, tmp) / tmp.shape[0]
    print("start to compute SVD")
    A, L, _ = linalg.svd(cov_mat)
    ZCA_mat = np.dot(A, np.dot(np.diag(1. / (np.sqrt(L) + 0.01)), A.T))
    try:
        np.save('zca_mat.npy', ZCA_mat)
    except:
        print("could not save ZCA_mat")


if __name__ == '__main__':
    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time)+" [sec]")
