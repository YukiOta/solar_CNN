# coding: utf-8
# Prediction with CNN

#libray# {{{
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import time
import os
import argparse
# matplotlib.use('Agg')
# from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential, model_from_json, model_from_yaml
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Adadelta, RMSprop
# from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import ndimage
import seaborn as sns

SAVE_dir = "./RESULT/CNN_CV_100/"
if not os.path.isdir(SAVE_dir):
    os.makedirs(SAVE_dir)


# Define Model
def CNN_model1(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*2 -> [FC -> RELU]*2 -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model2(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
    """
    INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation=activation))
    model.add(Dropout(0.5))

    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


def CNN_model3(
    activation="relu",
    loss="mean_squared_error",
    optimizer="Adadelta",
    layer=0,
    height=0,
    width=0):
    """
    INPUT -> [CONV -> RELU] -> OUT
    """
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(layer, height, width)))
    model.add(Convolution2D(16, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(1))

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

# target normilization


def norm_target(target):
    mms = MinMaxScaler()
    target = target.reshape(-1, 1)
    target_norm = mms.fit_transform(target)
    target_norm = target_norm.reshape((target.size))
    return target_norm


def data_plot(model, target, img, batch_size=10, date="hoge", save_csv=False):

    num = []
    time = []
    for i in range(target[0].shape[0]):
        if i % 50 == 0:
            num.append(i)
            time.append(int(target[0][i]))
        if i == target[0].shape[0] - 1:
            num.append(i)
            time.append(int(target[0][i]))
    img_ = img.transpose(0, 3, 1, 2).copy()
    pred = model.predict(img_, batch_size=batch_size, verbose=1).copy()
    if pred.shape:
        print(pred.shape)
    if type(pred):
        print(type(pred))
    print(target[1].shape)
    plt.figure()
    plt.plot(pred, label="Predicted")
    plt.plot(target[1], label="Observed")
    plt.legend(loc='best')
    plt.title("Prediction tested on"+date)
    plt.xlabel("Time")
    plt.ylabel("Generated Power[kW]")
    plt.ylim(0, 25)
    plt.xticks(num, time)

    pred_ = pred.reshape(pred.shape[0])
    if save_csv is True:
        # calculate_error(target[1], pred_, title=date, savefig=False)
        save_target_and_prediction(target=target[1], pred=pred_, title=date)

    filename = date + "_data"
    i = 0
    while os.path.exists(SAVE_dir+'{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir+'{}{:d}.png'.format(filename, i))


def loss_plot(hist, opt_name="hoge"):
    plt.figure()
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    nb_e = len(loss)
    plt.plot(range(nb_e), np.log(loss), "o-", label="loss")
    plt.plot(range(nb_e), np.log(val_loss), "o-", label="val_loss")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    filename = opt_name + "_loss"
    i = 0
    while os.path.exists(SAVE_dir+'{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir+'{}{:d}.png'.format(filename, i))

    # }}}


def acc_plot(hist, opt_name="hoge"):
    plt.figure()
    acc = hist.history['acc']
    val_acc = hist.history['val_acc']
    nb_e = len(acc)
    plt.plot(range(nb_e), np.log(acc), "o-", label="acc")
    plt.plot(range(nb_e), np.log(val_acc), "o-", label="val_acc")
    plt.legend(loc='best')
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("acc")

    filename = opt_name + "_acc"
    i = 0
    while os.path.exists(SAVE_dir+'{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir+'{}{:d}.png'.format(filename, i))


def save_model(model, opt_name):
    i = 0
    filename = opt_name
    while os.path.exists("{}{:d}.json".format(filename, i)):
        i += 1
    open('{}{:d}.json'.format(filename, i), 'w').write(model.to_json())
    model.save_weights('{}{:d}.h5'.format(filename, i), overwrite=True)


def predict_compare(model1, model2, model3):
    timedelta = dt.timedelta(seconds=30)
    t27_b = dt.datetime(year=2017, month=1, day=27, hour=12, minute=55, second=15)
    t26_b = dt.datetime(year=2017, month=1, day=26, hour=13, minute=30, second=15)
    dt_26 = [0, 48, 98, 148, 198, 248, 290]
    dt_27 = [0, 48, 98, 148, 198, 248, 299]
    dt_x26 = []
    dt_x27 = []

    for t in dt_26:
        dt_tmp = t26_b + timedelta * t
        dt_x26.append(dt_tmp)
    for t in dt_27:
        dt_tmp = t27_b + timedelta * t
        dt_x27.append(dt_tmp)
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    tmp1 = img_ts[0]
    tmp_1 = target_ts[0][1]
    prd1 = model1.predict(tmp1, batch_size=10, verbose=1)
    prd2 = model2.predict(tmp1, batch_size=10, verbose=1)
    prd3 = model3.predict(tmp1, batch_size=10, verbose=1)
    plt.plot(tmp_1, label="original(20170126)")
    plt.plot(prd1, label="predict_v1(20170126)")
    plt.plot(prd2, label="predict_v2(20170126)")
    plt.plot(prd3, label="predict_v3(20170126)")
    plt.legend(loc="best")
    plt.xlabel("time")
    plt.xticks([0, 48, 98, 148, 198, 248, 290],
               [dt_x26[0].strftime("%H:%M:%S"),
                dt_x26[1].strftime("%H:%M:%S"),
                dt_x26[2].strftime("%H:%M:%S"),
                dt_x26[3].strftime("%H:%M:%S"),
                dt_x26[4].strftime("%H:%M:%S"),
                dt_x26[5].strftime("%H:%M:%S"),
                dt_x26[6].strftime("%H:%M:%S")])
    plt.ylabel("Generated Power[kW]")
    plt.tight_layout()

    plt.subplot(2,1,2)
    tmp2 = img_ts[1]
    tmp_2 = target_ts[1][1]
    prd4 = model1.predict(tmp2, batch_size=10, verbose=1)
    prd5 = model2.predict(tmp2, batch_size=10, verbose=1)
    prd6 = model3.predict(tmp2, batch_size=10, verbose=1)
    plt.plot(tmp_2, label="original(20170127)")
    plt.plot(prd4, label="predict_v1(20170127)")
    plt.plot(prd5, label="predict_v2(20170127)")
    plt.plot(prd6, label="predict_v3(20170127)")
    plt.xlabel("time")
    plt.xticks([0, 48, 98, 148, 198, 248, 299],
               [dt_x27[0].strftime("%H:%M:%S"),
                dt_x27[1].strftime("%H:%M:%S"),
                dt_x27[2].strftime("%H:%M:%S"),
                dt_x27[3].strftime("%H:%M:%S"),
                dt_x27[4].strftime("%H:%M:%S"),
                dt_x27[5].strftime("%H:%M:%S"),
                dt_x27[6].strftime("%H:%M:%S")])
    plt.ylabel("Generated Power[kW]")
    plt.tight_layout()
    plt.savefig(SAVE_dir+"predit_compare.pdf")

#
# def calculate_error(y1_true, y1_pred, title='title', savefig=False):
#
#     MAE1 = mean_absolute_error((y1_true, y1_true), (y1_pred, y1_pred), multioutput='raw_values')
#     RMSE1 = mean_squared_error((y1_true, y1_true), (y1_pred, y1_pred), multioutput='raw_values')
#     df1 = pd.DataFrame(MAE1)
#     df2 = pd.DataFrame(RMSE1)
#     df = pd.concat([df1, df2], axis=1)
#     df.columns = ["MAE", "RMSE"]
#     df.to_csv(SAVE_dir+"Error_csv_"+title+".csv")
#     # plt.figure()
#     # ax = sns.boxplot(data=df)
#     # plt.title("MAE and RMSE errors")
#     # plt.xlabel("Metrics")
#     # plt.ylabel("Error [kW]")
#     if savefig is True:
#         plt.savefig(SAVE_dir+"Error_plot"+title+".png")
#

def save_target_and_prediction(target, pred, title):

    pred_df = pd.DataFrame(pred)
    target_df = pd.DataFrame(target)
    df = pd.concat([target_df, pred_df], axis=1)
    df.columns = ["TARGET", "PREDICTION"]
    df.to_csv(SAVE_dir+"Error_csv_"+title+".csv")


def show_image(img, title="hoge"):
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.show()

def compute_mean(image_array):
    """全画像の平均をとって、平均を返す
    入力：画像データセット (np配列を想定してる) [枚数, height, width, rgb]
    出力：平均画像
    """
    print("conmpute mean image")
    mean_image = np.ndarray.mean(image_array, axis=0)
    return mean_image


def main():
    # functions
    # read images
    image2016_tmp = np.load(DATA_DIR+'images2016_resized100.npz')
    image2017_tmp = np.load(DATA_DIR+'images2017_resized100.npz')
    # image2016_tmp = np.load('./images2016_resized100.npz')
    # image2017_tmp = np.load("./images2017_resized100.npz")
    target2016_tmp = np.load(TARGET_DIR+'target2016.npz')
    target2017_tmp = np.load(TARGET_DIR+'target2017.npz')

    # target_20170126 = load_target(CSV_dir+"power_data20170126.csv")
    # target_20170127 = load_target(CSV_dir+"power_data20170127.csv")
    # img_tmp = np.load("./solardata_2017.npz")

    img_tr = []
    target_tr = []

    print("Loading Images...")

    img_tr.append(image2016_tmp['x1'])
    img_tr.append(image2016_tmp['x2'])
    img_tr.append(image2016_tmp['x3'])
    img_tr.append(image2016_tmp['x4'])
    img_tr.append(image2017_tmp['x1'])
    img_tr.append(image2017_tmp['x2'])
    img_tr.append(image2017_tmp['x3'])
    img_tr.append(image2017_tmp['x4'])  # 7
    img_tr.append(image2017_tmp['x5'])
    img_tr.append(image2017_tmp['x6'])
    img_tr.append(image2017_tmp['x7'])
    img_tr.append(image2017_tmp['x8'])
    img_tr.append(image2017_tmp['x9'])

########temporal space
    for i in range(len(img_tr)):
        img_tr[i] = ndimage.median_filter(img_tr[i], 3)


#######

    target_tr.append(target2016_tmp['y1'])
    target_tr.append(target2016_tmp['y2'])
    target_tr.append(target2016_tmp['y3'])
    target_tr.append(target2016_tmp['y4'])
    target_tr.append(target2017_tmp['y1'])
    target_tr.append(target2017_tmp['y2'])
    target_tr.append(target2017_tmp['y3'])
    target_tr.append(target2017_tmp['y4'])
    target_tr.append(target2017_tmp['y5'])
    target_tr.append(target2017_tmp['y6'])
    target_tr.append(target2017_tmp['y7'])
    target_tr.append(target2017_tmp['y8'])
    target_tr.append(target2017_tmp['y9'])
    print('done')

    date_list = ["2016_11_17",
                 "2016_11_30",
                 "2016_12_18",
                 "2016_12_21",
                 "2017_01_12",
                 "2017_01_14",
                 "2017_01_15",
                 "2017_01_18",
                 "2017_01_22",
                 "2017_01_23",
                 "2017_01_26",
                 "2017_01_27",
                 "2017_01_30"]

    weather_list = [
        "Cloud",
        "Cloud",
        "Sunny",
        "Sunny",
        "Sunny",
        "Cloud",
        "Cloud",
        "Cloud",
        "Sunny",
        "Sunny",
        "Sunny",
        "Cloud",
        "Cloud"]
    test_error_list = []

    for i in range(13):

        ts_img = img_tr.pop(i)
        ts_target = target_tr.pop(i)

        img_tr_all = np.concatenate((
            img_tr[0], img_tr[1], img_tr[2], img_tr[3], img_tr[4],
            img_tr[5], img_tr[6], img_tr[7], img_tr[8], img_tr[9], img_tr[10],
            img_tr[11]
        ), axis=0)

        target_tr_all = np.concatenate((
            target_tr[0], target_tr[1], target_tr[2], target_tr[3], target_tr[4],
            target_tr[5], target_tr[6], target_tr[7], target_tr[8], target_tr[9],
            target_tr[10], target_tr[11]
        ), axis=1)

        mean_img = compute_mean(image_array=img_tr_all)
        img_tr_all -= mean_img
        ts_img -= mean_img

        # transpose for CNN INPUT shit
        img_tr_all = img_tr_all.transpose(0, 3, 1, 2)
        print(img_tr_all.shape)
        # set image size
        layer = img_tr_all.shape[1]
        height = img_tr_all.shape[2]
        width = img_tr_all.shape[3]

        print("Image and Target Ready")

        # parameter
        activation = ["relu", "sigmoid"]
        optimizer = ["adam", "adadelta", "rmsprop"]
        nb_epoch = [10, 25, 50]
        batch_size = [5, 10, 15]

        # model set
        model = None
        model = CNN_model3(
            activation="relu",
            optimizer="Adadelta",
            layer=layer,
            height=height,
            width=width)
        # plot_model(model, to_file='CNN_model.png')

        # initialize check
        data_plot(
            model=model, target=ts_target, img=ts_img, batch_size=10,
            date=date_list[i], save_csv=True)

        early_stopping = EarlyStopping(patience=3, verbose=1)

        # Learning model
        hist = model.fit(img_tr_all, target_tr_all[1],
                         nb_epoch=nb_epoch[0],
                         batch_size=batch_size[1],
                         validation_split=0.1,
                         callbacks=[early_stopping])
        data_plot(
            model=model, target=ts_target, img=ts_img, batch_size=10,
            date=date_list[i], save_csv=True)
        # evaluate
        try:
            img_tmp = ts_img.transpose(0, 3, 1, 2)
            score = model.evaluate(img_tmp, ts_target[:, 1], verbose=1)
            print("Evaluation "+date_list[i])
            print('TEST LOSS: ', score[0])
            test_error_list.append(score[0])
        except:
            print("error in evaluation")

        # put back data
        img_tr.insert(i, ts_img)
        target_tr.insert(i, ts_target)

    with open(SAVE_dir+"test_loss.txt", "w") as f:
        f.write(str(test_error_list))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../data_old/",
        help="choose your data (image) directory"
    )
    parser.add_argument(
        "--target_dir",
        default="../data_old/",
        help="choose your target dir"
    )
    args = parser.parse_args()
    DATA_DIR, TARGET_DIR = args.data_dir, args.target_dir

    start = time.time()
    main()
    elapsed_time = time.time() - start
    print("elapsed_time:{0}".format(elapsed_time)+" [sec]")


######################################################
# DATA_DIR = "../data_old/"
# TARGET_DIR = "../data_old/"



















# end
