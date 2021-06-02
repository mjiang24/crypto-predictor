
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pickle

data = pickle.load(open("data.pickle", "rb"))

BATCH_SIZE = data["batch_size"]
EPOCHS = data["epochs"]
NAME = data["name"] + str(int(time.time()))

train_x = data["train_x"]
train_y = data["train_y"]

validation_x = data["validation_x"]
validation_y = data["validation_y"]

model = Sequential()

model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128,return_sequences = True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation = "relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation ="softmax"))

opt = tf.keras.optimizers.Adam(lr = 0.001, decay = 1e-6)

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = opt, metrics = ['accuracy'])

tensorboard = TensorBoard(log_dir = f'logs/{NAME}')
filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"
checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (validation_x, validation_y), callbacks = [tensorboard, checkpoint])