# codeing: utf-8
"""CNNで吐き出したCSVファイルを使って、結果を吟味します。
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
sns.set(style="whitegrid")

def plot_box(data_ori, data_con, date="hoge"):

    dif_ori = data_ori[:, 1] - data_ori[:, 2]  # target - prediction
    name_ori = np.array(["original"]*len(dif_ori))
    day = np.array([date]*len(dif_ori))
    df_dif = pd.DataFrame(dif_ori)
    df_name = pd.DataFrame(name_ori)
    df_day = pd.DataFrame(day)

    df_ori = pd.concat([df_dif, df_name, df_day], axis=1)
    df_ori.columns = ["dif", "type", "date"]

    dif_con = data_con[:, 1] - data_con[:, 2]  # target - prediction
    name_con = np.array(["proposed"]*len(dif_con))
    day = np.array([date]*len(dif_con))
    df_dif = pd.DataFrame(dif_con)
    df_name = pd.DataFrame(name_con)
    df_day = pd.DataFrame(day)

    df_con = pd.concat([df_dif, df_name, df_day], axis=1)
    df_con.columns = ["dif", "type", "date"]

    df = pd.concat([df_ori, df_con], axis=0)

    plt.figure(figsize=(8, 8))
    sns.boxplot(x="date", y="dif", hue="type", data=df)
    # sns.despine(offset=10, trim=True)
    plt.show()



data_origin = "./RESULT/multiple_bands/CNN_keras_multiple_bands_100/"
data_consider = "./RESULT/multiple_bands/CNN_keras_multiple_bands_considered100/"
data1 = np.loadtxt(data_origin+"Error_csv_20170503.csv", delimiter=",", skiprows=1, dtype=float)
data2 = np.loadtxt(data_consider+"Error_csv_20170503.csv", delimiter=",", skiprows=1, dtype=float)

data_origin_list = os.listdir(data_origin)
data_consider_list = os.listdir(data_consider)

for filename in data_origin_list:
    if filename.startswith("Error"):
        data_ori = np.loadtxt(data_origin+filename, delimiter=",", skiprows=1, dtype=float)
        data_con = np.loadtxt(data_consider+filename, delimiter=",", skiprows=1, dtype=float)
        plot_box(data_ori, data_con, date=filename[10:18])

























# end
