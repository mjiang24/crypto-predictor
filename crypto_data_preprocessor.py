import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import pickle
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

SEQ_LEN = 60
FUTURE_PERIOD_PREDICT = 3
PRICE_TO_PREDICT = "LTC-USD"


NAME = str(SEQ_LEN) + "-SEQ-" + str(FUTURE_PERIOD_PREDICT)+ "-PRED-"
EPOCHS = 10
BATCH_SIZE =64

def classify(current, future):
    if float(future) > float(current):
        return 1
    else:
        return 0

def preprocess_df(df):
    df = df.drop("future",1)
    for col in df.columns:
        if col != "target":
            #normalize data to percentage change
            df[col] = df[col].pct_change()
            df.dropna(inplace=True)
            df[col] = preprocessing.scale(df[col].values)
    df.dropna(inplace = True)

    sequential_data = []
    prev_days = deque(maxlen = SEQ_LEN)

    for i in df.values:
        prev_days.append([n for n in i[:-1]])
        if len(prev_days) == SEQ_LEN:
            sequential_data.append([np.array(prev_days), i[-1]])

    random.shuffle(sequential_data)

    buys = []
    sells = []

    for seq, target in sequential_data:
        if target==0:
            sells.append([seq, target])
        elif target ==1:
            buys.append([seq, target])

    random.shuffle(buys)
    random.shuffle(sells)

    #balancing number of buys and sells
    smaller = min(len(buys), len(sells))

    buys = buys[:smaller]
    sells = sells[:smaller]

    sequential_data = buys + sells

    random.shuffle(sequential_data)

    X = []
    y = []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)



main_df = pd.DataFrame()

prices = ["BTC-USD", "LTC-USD", "ETH-USD", "BCH-USD"]

for price_ratio in prices:

    dataset = "crypto_data/" + price_ratio + ".csv"


    df = pd.read_csv(dataset, names = ["time", "low", "high","open", "close","volume"])

    df.rename(columns = {"close":price_ratio+"_close", "volume": price_ratio+"_volume"}, inplace = True)

    df.set_index("time", inplace = True)
    df = df[[price_ratio+"_close",price_ratio+"_volume"]]

    if len(main_df) == 0:
        main_df = df
    else:
        main_df = main_df.join(df)

main_df['future'] = main_df[PRICE_TO_PREDICT+"_close"].shift(-FUTURE_PERIOD_PREDICT)


main_df['target'] = list(map(classify, main_df[PRICE_TO_PREDICT+"_close"], main_df["future"]))

#print(main_df[[PRICE_TO_PREDICT+"_close", "future", "target"]].head())

times = sorted(main_df.index.values)
last_5pct = times[-int(0.05*len(main_df))]

#splitting data into training and validation
validation_main_df = main_df[(main_df.index >= last_5pct)]
main_df = main_df[(main_df.index < last_5pct)]

train_x, train_y = preprocess_df(main_df)
validation_x, validation_y = preprocess_df(validation_main_df)

pickle_out = open("data.pickle","wb")
pickle.dump({"epochs": EPOCHS, "batch_size" : BATCH_SIZE ,"name" : NAME, "train_x": train_x, "train_y" :train_y, "validation_x" :validation_x, "validation_y" :validation_y}, pickle_out)
pickle_out.close()




