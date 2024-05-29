#!/usr/bin/env python3

import os, time, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fp = '{}/dev-repo/tf-scaffold/images/mnist/img_{}.jpg'.format(os.getenv('HOME'), int(time.time()))

(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.boston_housing.load_data()
train_data.shape
#(404, 13)
test_data.shape
#(102, 13)

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
 model = tf.keras.models.Sequential()
 model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
 model.add(tf.keras.layers.Dense(64, activation='relu'))
 model.add(tf.keras.layers.Dense(1))
 model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01), loss=tf.keras.losses.mse, metrics=['mae'])
 return model

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_score = []
for i in range(k):
 print('processing fold #', i)
 val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
 val_labels = train_labels[i*num_val_samples : (i+1)*num_val_samples]
 partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
 partial_train_labels = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i+1)*num_val_samples:]], axis=0)
 model = build_model()
 model.fit(partial_train_data, partial_train_labels, epochs=num_epochs, batch_size=16, verbose=0)
 val_mse, val_mae = model.evaluate(val_data, val_labels, verbose=0)
 all_score.append(val_mae)

all_score
#[7.945329189300537, 39.825138092041016, 15.435348510742188, 10.13697338104248]
np.mean(all_score)
#18.335697293281555

num_epochs = 500
all_mae_histories = []
for i in range(k):
 print('processing fold #', i)
 val_data = train_data[i*num_val_samples : (i+1)*num_val_samples]
 val_labels = train_labels[i*num_val_samples : (i+1)*num_val_samples]
 partial_train_data = np.concatenate([train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0)
 partial_train_labels = np.concatenate([train_labels[:i*num_val_samples], train_labels[(i+1)*num_val_samples:]], axis=0)
 model = build_model()
 history = model.fit(partial_train_data, partial_train_labels, epochs=num_epochs, validation_data=(val_data, val_labels), batch_size=16, verbose=0)
 #history.history.keys()
 all_mae_histories.append(history.history['val_mae'])

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('{}/dev-repo/tf-scaffold/images/boston_housing/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()

def smooth_curve(points, factor=0.9):
 smoothed_points = []
 for point in points:
  if smoothed_points:
    previous = smoothed_points[-1]
    smoothed_points.append(previous * factor + point * (1 - factor))
  else:
    smoothed_points.append(point)
 return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.savefig('{}/dev-repo/tf-scaffold/images/boston_housing/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()


# train and evaluate with the whole dataset
model = build_model()
model.fit(train_data, train_labels, epochs=80, batch_size=16)
#Epoch 80/80
#26/26 [==============================] - 0s 5ms/step - loss: 5.4830 - mae: 1.7407
model.evaluate(test_data, test_labels)
#[13.394491195678711, 2.4863851070404053]
model.metrics_names
#['loss', 'mae']
