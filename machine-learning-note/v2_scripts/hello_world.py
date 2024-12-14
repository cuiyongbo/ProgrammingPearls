#!/usr/bin/env python

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

class MyModel(tf.keras.Model):
 def __init__(self):
  super(MyModel, self).__init__()
  self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
  self.flatten = tf.keras.layers.Flatten()
  self.d1 = tf.keras.layers.Dense(128, activation='relu')
  self.d2 = tf.keras.layers.Dense(10, activation='softmax')
 def call(self, x):
  x = self.conv1(x)
  x = self.flatten(x)
  x = self.d1(x)
  return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
 with tf.GradientTape() as tape:
  pred = model(images, training=True)
  loss = loss_object(labels, pred)
 gradients = tape.gradient(loss, model.trainable_variables)
 optimizer.apply_gradients(zip(gradients, model.trainable_variables))
 train_loss(loss)
 train_accuracy(labels, pred)

@tf.function
def test_step(images, labels):
 pred = model(images, training=False)
 loss = loss_object(labels, pred)
 test_loss(loss)
 test_accuracy(labels, pred)

EPOCHS = 5
for epoch in range(EPOCHS):
 train_loss.reset_states()
 train_accuracy.reset_states()
 test_loss.reset_states()
 test_accuracy.reset_states()
 for images, labels in train_ds:
  train_step(images, labels)
 for images, labels in test_ds:
  test_step(images, labels)
 print(f'Epoch {epoch+1}, Loss: {train_loss.result()}, Accuracy: {train_accuracy.result()*100}, Test Loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()*100}')