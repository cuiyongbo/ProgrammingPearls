#!/usr/bin/env python3

# train a naive model similar to https://tensorflow.google.cn/tutorials/keras/classification

import os
import tensorflow as tf
import matplotlib.pyplot as plt
print("TensorFlow version:", tf.__version__)

epochs = 10
batch_size = 64
num_units = 128
num_classes = 10
num_features = 28*28
drop_out_rate = 0.2
learning_rate = 0.01

# preprocess input
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# save a test image for later usage
plt.imsave(fname='./data/mnist_test_demo.jpg', arr=x_test[0])
x_train, x_test = x_train.reshape(-1, num_features) / 255.0, x_test.reshape(-1, num_features) / 255.0

# define a model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(num_units, activation='relu', input_shape=(num_features,)))
model.add(tf.keras.layers.Dropout(drop_out_rate))
model.add(tf.keras.layers.Dense(num_classes))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# training the defined model
model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)

# evalute model performance
model.evaluate(x_test, y_test)

# serialize model to disk for later usage
saved_model_dir = './data/mnist/1'
tf.saved_model.save(model, saved_model_dir)
# restore model
# model = tf.saved_model.load(saved_model_dir)

os.listdir(saved_model_dir)
#['variables', 'saved_model.pb', 'assets']