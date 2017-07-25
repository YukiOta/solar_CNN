# coding: utf-8
"""solar_projectにおける、データ整理用プログラム
1.発電量csvファイルから、必要な列を取り出して、整理する
2.画像の読み込み
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import datetime as dt
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


def load_target(csv, imgdir, interval):
    """
    csvファイルを読み込む
    パンダスを用いて、処理する
    csv: csvファイルのありか
    imgdir: 対応するimgdirのありか
    interval: 撮影インターバル (sec)
    """
    # csvファイルの読み込み
    df = pd.read_csv(csv)

    # 必要な列を取り出し、必要のない行を取り除く
    df = df.iloc[:, [0, 11, 36]]
    df.columns = ["time", "power", "temperature"]
    df = df.drop([0, 1])
    # データフレームをnumpy配列にする
    tmptmp = np.array(df)

    # 画像撮影時刻を補正する
    filelist = os.listdir(imgdir)
    time_start, time_end = check_target_time(filelist=filelist)

    #
    for i in range(len(tmptmp)):

        if tmptmp[i][0].replace(':', '') == time_start:
            i_start = i
            # print(i)
            # print(tmptmp[i])
        elif tmptmp[i][0].replace(':', '') == time_end:
            i_end = i
            # print(i)
            # print(tmptmp[i])
    interval = 60
    ota = tmptmp[i_start:i_end:int(interval/6)]

    target = []
    target.append(df.values[:, 0])
    target.append(df.values[:, 1])
    target = np.array(target, np.float64)
    print("CSV Load Done")
    return target


def check_target_time(filelist):
    """
    画像の時刻と、ファイルの時刻が一致しないことがあるので、チェックする。
    """
    file_list = filelist
    time_tmp = file_list[0][9:15]
    if int(time_tmp[-2:]) % 6 >= 3:
        delta = 6 - (int(time_tmp) % 6)
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start + timedelta
        time_correct_s = time_correct.strftime("%H%M%S")
    elif int(time_tmp[-2:]) % 6 < 3:
        delta = int(time_tmp[-2:]) % 6
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start - timedelta
        time_correct_s = time_correct.strftime("%H%M%S")

    time_tmp = file_list[-2][9:15]
    if int(time_tmp[-2:]) % 6 >= 3:
        delta = 6 - (int(time_tmp) % 6)
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start + timedelta
        time_correct_e = time_correct.strftime("%H%M%S")
    elif int(time_tmp[-2:]) % 6 < 3:
        delta = int(time_tmp[-2:]) % 6
        timedelta = dt.timedelta(seconds=delta)
        time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
        time_correct = time_start - timedelta
        time_correct_e = time_correct.strftime("%H%M%S")

    return time_correct_s, time_correct_e


df = pd.read_csv("/Users/yukiota/solar_project/data/PV_CSV/201611/Trd20161117_1.csv")
df = df.iloc[:, [0, 11, 36]]
df.columns = ["time", "power", "temperature"]
df = df.drop([0, 1])
df[7500:8000]
file_list = os.listdir("/Users/yukiota/solar_project/data/PV_IMAGE/201611/20161117_IT_resampled_northup")
file_list[0][9:15]
file_list[-2][9:15]
tmptmp = np.array(df)

time_tmp = file_list[0][9:15]
if int(time_tmp[-2:]) % 6 >= 3:
    delta = 6 - (int(time_tmp) % 6)
    timedelta = dt.timedelta(seconds=delta)
    time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
    time_correct = time_start + timedelta
    time_correct_s = time_correct.strftime("%H%M%S")
elif int(time_tmp[-2:]) % 6 < 3:
    delta = int(time_tmp[-2:]) % 6
    timedelta = dt.timedelta(seconds=delta)
    time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
    time_correct = time_start - timedelta
    time_correct_s = time_correct.strftime("%H%M%S")

time_tmp = file_list[-2][9:15]
if int(time_tmp[-2:]) % 6 >= 3:
    delta = 6 - (int(time_tmp) % 6)
    timedelta = dt.timedelta(seconds=delta)
    time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
    time_correct = time_start + timedelta
    time_correct_e = time_correct.strftime("%H%M%S")
elif int(time_tmp[-2:]) % 6 < 3:
    delta = int(time_tmp[-2:]) % 6
    timedelta = dt.timedelta(seconds=delta)
    time_start = dt.datetime.strptime(time_tmp, "%H%M%S")
    time_correct = time_start - timedelta
    time_correct_e = time_correct.strftime("%H%M%S")


for i in range(len(tmptmp)):

    if tmptmp[i][0].replace(':', '') == time_correct_s:
        i_start = i
        print(i)
        print(tmptmp[i])
    elif tmptmp[i][0].replace(':', '') == time_correct_e:
        i_end = i
        print(i)
        print(tmptmp[i])
interval = 60
ota = tmptmp[i_start:i_end:int(interval/6)]
ota.shape


len(file_list[0:-2])















# end
