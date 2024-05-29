#!/usr/bin/env python3

import os, time, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#fp = '{}/dev-repo/tf-scaffold/images/mnist/img_{}.jpg'.format(os.getenv('HOME'), int(time.time()))

timestamps = 100
input_features = 32
output_features = 64
inputs = np.random.random((timestamps, input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
for input_t in inputs:
 output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
 successive_outputs.append(output_t)
 state_t = output_t

final_output_sequence = np.concatenate(successive_outputs, axis=0)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10000, 32))
model.add(tf.keras.layers.SimpleRNN(32)) # return only the last output
model.summary()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10000, 32))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True)) # return all past-time outputs
model.summary()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10000, 32))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(32, return_sequences=True))
model.add(tf.keras.layers.SimpleRNN(32))
model.summary()


# Use the default parameters to keras.datasets.imdb.load_data
start_char = 1
oov_char = 2
index_from = 3
# Retrieve the training sequences.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(start_char=start_char, oov_char=oov_char, index_from=index_from, num_words=10000)
# Retrieve the word index file mapping words to indices
word_index = tf.keras.datasets.imdb.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
# And add `index_from` to indices to sync with `x_train`
inverted_word_index = dict((i + index_from, word) for (word, i) in word_index.items())
# Update `inverted_word_index` to include `start_char` and `oov_char`
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"
# Decode the first sequence in the dataset
decoded_sequence = " ".join(inverted_word_index[i] for i in x_train[0])

maxlen = 500
batch_size = 32
max_features = 10000

# reverse sequences to test bidirectional rnn
# x_train = [x[::-1] for x in x_train]
# x_test = [x[::-1] for x in x_test]

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
# x_train.shape
# (25000, 500)
# x_test.shape
# (25000, 500)

# simple rnn solution suffers from vanishing-gradient problem
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_features, 32))
model.add(tf.keras.layers.SimpleRNN(32))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

# lstm solution
model = tf.keras.models.Sequential()                                                   
model.add(tf.keras.layers.Embedding(max_features, 32))
model.add(tf.keras.layers.LSTM(32))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)
# Epoch 10/10
# 625/625 [==============================] - 71s 113ms/step - loss: 0.1328 - acc: 0.9539 - val_loss: 0.2995 - val_acc: 0.8940
model.evaluate(x_test, y_test)
# 782/782 [==============================] - 43s 55ms/step - loss: 0.3502 - acc: 0.8743
# [0.3502245843410492, 0.8743200302124023]

# result of reversed squences
# Epoch 10/10
# 157/157 [==============================] - 32s 202ms/step - loss: 0.1650 - acc: 0.9465 - val_loss: 0.3675 - val_acc: 0.8660
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 45s 57ms/step - loss: 2.7418 - acc: 0.5216
# [2.7417712211608887, 0.5216400623321533]

# bidirectional RNN
model = tf.keras.models.Sequential()                                                   
model.add(tf.keras.layers.Embedding(max_features, 32))
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=batch_size, validation_split=0.2)
# Epoch 10/10
# 625/625 [==============================] - 913s 1s/step - loss: 0.1394 - acc: 0.9520 - val_loss: 0.3132 - val_acc: 0.8880
model.evaluate(x_test, y_test)
# 782/782 [==============================] - 1203s 2s/step - loss: 0.3683 - acc: 0.8703
# [0.3682785928249359, 0.8703200817108154]


# result of reversed squences
# Epoch 10/10
# 157/157 [==============================] - 55s 353ms/step - loss: 0.1599 - acc: 0.9487 - val_loss: 0.3552 - val_acc: 0.8706
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 88s 112ms/step - loss: 2.3080 - acc: 0.5051
# [2.3080317974090576, 0.505120038986206]



def plot_acc_and_loss(history):
 acc = history.history['acc']
 val_acc = history.history['val_acc']
 loss = history.history['loss']
 val_loss = history.history['val_loss']
 epochs = range(1, len(acc) + 1)
 plt.plot(epochs, acc, 'bo', label='Training acc')
 plt.plot(epochs, val_acc, 'b', label='Validation acc')
 plt.title('Training and validation accuracy')
 plt.legend()
 plt.savefig('{}/dev-repo/tf-scaffold/images/imdb/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
 plt.show()
 plt.figure()
 plt.plot(epochs, loss, 'bo', label='Training loss')
 plt.plot(epochs, val_loss, 'b', label='Validation loss')
 plt.title('Training and validation loss')
 plt.legend()
 plt.savefig('{}/dev-repo/tf-scaffold/images/imdb/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
 plt.show()


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


# using rnn to solve weather-forcasting problem
# wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip
data_dir = os.path.join(os.getenv('HOME'), '.keras/datasets/jena_climate')
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
with open(fname) as f:
 data = f.read()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]


float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
 values = [float(x) for x in line.split(',')[1:]]
 float_data[i, :] = values

#float_data.shape
#(420451, 14)

temp = float_data[:, 1]
plt.plot(range(len(temp)), temp)
plt.savefig('{}/dev-repo/tf-scaffold/images/weather_forecasting/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()


mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
 if max_index is None:
  max_index = len(data) - delay - 1
 i = min_index + lookback
 while True:
  if shuffle:
   rows = np.random.randint(min_index+lookback, max_index, size=batch_size)
  else:
   if i+batch_size >= max_index:
    i = min_index + lookback
   rows = np.arange(i, min(i+batch_size, max_index))
   i+=len(rows)
  samples = np.zeros((len(rows), lookback//step, data.shape[-1]))
  targets = np.zeros((len(rows),))
  for j, row in enumerate(rows):
   indices = range(rows[j]-lookback, rows[j], step)
   samples[j] = data[indices]
   targets[j] = data[rows[j] + delay][1]
  yield samples, targets



def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets



# float_data.shape
# (420451, 14)
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0, max_index=200000, shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=200001, max_index=300000, shuffle=True, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=300001, max_index=None, shuffle=True, step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback)
test_steps = (len(float_data) - 300001 - lookback)

def evaluate_naive_method():
 batch_maes = []
 for step in range(val_steps):
  samples, targets = next(val_gen)
  pred = samples[:, -1, 1]
  mae = np.mean(np.abs(pred - targets))
  batch_maes.append(mae)
  print(np.mean(batch_maes))
  return np.mean(batch_maes)

a = evaluate_naive_method()
0.25664539514803253
a * std[0]
2.1764643054071926

# training stalls at the first epoch
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(lookback//step, float_data.shape[-1])))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.mae)
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)


# add GRU layer, still suffering from training stall
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(32, input_shape=(None, float_data.shape[-1])))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.mae)
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20, validation_data=val_gen, validation_steps=val_steps)

# add dropout to fight overfit, training stalls
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, float_data.shape[-1])))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.mae)
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

# stacking recurrent layers, training stalls
model = tf.keras.Sequential()
model.add(tf.keras.layers.GRU(32, dropout=0.1, recurrent_dropout=0.5, return_sequences=True, input_shape=(None, float_data.shape[-1])))
model.add(tf.keras.layers.GRU(32, dropout=0.1, recurrent_dropout=0.5))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss=tf.keras.losses.mae)
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40, validation_data=val_gen, validation_steps=val_steps)

# bidirectional rnn, training stalls
model = tf.keras.Sequential()
model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(16, input_shape=(None, float_data.shape[-1]))))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='rmsprop', loss=tf.keras.losses.mae)
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=10, validation_data=val_gen, validation_steps=val_steps)
