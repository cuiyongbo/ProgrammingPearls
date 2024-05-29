#!/usr/bin/env python3

import tensorflow as tf
import mnist_reader
import sys

train_images, train_labels = mnist_reader.load_mnist('MNIST_data/origin', kind='train')
test_images, test_labels = mnist_reader.load_mnist('MNIST_data/origin', kind='t10k')

print("train set: ", train_images.shape, ", length: ", len(train_labels))
print("test set: ", test_images.shape, ", length: ", len(test_labels))
#sys.exit(0)

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

model.evaluate(test_images, test_labels)

