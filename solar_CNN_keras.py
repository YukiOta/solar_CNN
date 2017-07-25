# coding: utf-8
""" Prediction with CNN
input: fisheye image
out: Generated Power
クロスバリデーションもする
とりあえずkeras
"""

# library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as import pdb
import datetime as dt
import os
import sys
import seaborn as sns
import glob
matplotlib.use('Agg')

SAVE_dir = "./RESULT/CNN_keras/"
if not os.path.isdir(SAVE_dir):
    os.makedirs(SAVE_dir)

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


def CNN_model3(activation="relu", loss="mean_squared_error", optimizer="Adadelta"):
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
        save_target_and_prediction(target=target, pred=pred_, title=date)

    filename = date + "_data"
    i = 0
    while os.path.exists(SAVE_dir+'{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig(SAVE_dir+'{}{:d}.png'.format(filename, i))


def save_target_and_prediction(target, pred, title):

    pred_df = pd.DataFrame(pred)
    target_df = pd.DataFrame(target)
    df = pd.concat([target_df, pred_df], axis=1)
    df.columns = ["TARGET", "PREDICTION"]
    df.to_csv(SAVE_dir+"Error_csv_"+title+".csv")


def main():








if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir",
        default="../data/",
        help="choose your data (image) directory"
    )
    parser.add_argument(
        "--target_dir",
        default="../data/",
        help="choose your target dir"
    )
    args = parser.parse_args()
    DATA_DIR, TARGET_DIR = args.data_dir, args.target_dir
    main()






os.path.exists(".py")
glob.glob(os.getcwd()+"/*.py")




















# end
