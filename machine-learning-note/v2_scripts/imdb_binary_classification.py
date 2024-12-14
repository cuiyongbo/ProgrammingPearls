#!/usr/bin/env python3

import os, time, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

fp = '{}/dev-repo/tf-scaffold/images/mnist/img_{}.jpg'.format(os.getenv('HOME'), int(time.time()))

# Use the default parameters to keras.datasets.imdb.load_data
start_char = 1
oov_char = 2
index_from = 3
# Retrieve the training sequences.
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.imdb.load_data(
    start_char=start_char, oov_char=oov_char, index_from=index_from, num_words=10000
)
# Retrieve the word index file mapping words to indices
word_index = tf.keras.datasets.imdb.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
# And add `index_from` to indices to sync with `x_train`
inverted_word_index = dict(
    (i + index_from, word) for (word, i) in word_index.items()
)
# Update `inverted_word_index` to include `start_char` and `oov_char`
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"
# Decode the first sequence in the dataset
decoded_sequence = " ".join(inverted_word_index[i] for i in train_data[0])


def vectorize_sequences(seqs, dims=10000):
 res = np.zeros((len(seqs), dims))
 for i, s in enumerate(seqs):
  res[i, s] = 1.
 return res


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.binary_crossentropy, metrics=[tf.keras.metrics.binary_accuracy])


x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history=model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#Epoch 20/20
#30/30 [==============================] - 0s 11ms/step - loss: 0.0059 - accuracy: 0.9985 - val_loss: 0.6917 - val_accuracy: 0.8647

model.evaluate(x_test, y_test)
#782/782 [==============================] - 3s 3ms/step - loss: 0.7517 - accuracy: 0.8528
#[0.7517000436782837, 0.8528000712394714]


history_dict = history.history

#history_dict.keys()
#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

plt.clf()
loss_val = history_dict['loss']
val_loss_val = history_dict['val_loss']
epochs = range(1, len(loss_val)+1)
plt.plot(epochs, loss_val, 'bo', label='Trainning_loss')
plt.plot(epochs, val_loss_val, 'b', label='Validation_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('{}/dev-repo/tf-scaffold/images/imdb/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
epochs = range(1, len(loss_val)+1)
plt.plot(epochs, acc, 'bo', label='Trainning_accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('{}/dev-repo/tf-scaffold/images/imdb/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()


# fewer training loop
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
#Epoch 4/4
#49/49 [==============================] - 0s 7ms/step - loss: 0.1698 - accuracy: 0.9397
model.evaluate(x_test, y_test)    
#782/782 [==============================] - 3s 4ms/step - loss: 0.2901 - accuracy: 0.8857
#[0.2901034951210022, 0.8857200741767883]
model.predict(x_test)
#array([[0.21012756],
#       [0.9999075 ],
#       [0.9325789 ],
#       ...,
#       [0.10360444],
#       [0.07331457],
#       [0.5513264 ]], dtype=float32)

# fewer hidden layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
#Epoch 4/4
#49/49 [==============================] - 0s 7ms/step - loss: 0.1814 - accuracy: 0.9378
model.evaluate(x_test, y_test)    
#782/782 [==============================] - 3s 3ms/step - loss: 0.2833 - accuracy: 0.8871
#[0.2833467125892639, 0.8871200680732727]

# more hidden layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=512)
#Epoch 4/4
#49/49 [==============================] - 0s 7ms/step - loss: 0.1814 - accuracy: 0.9378
model.evaluate(x_test, y_test)
#782/782 [==============================] - 3s 4ms/step - loss: 0.3195 - accuracy: 0.8746
#[0.3194538354873657, 0.8745600581169128]


# more hidden units per layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
#Epoch 4/4
#30/30 [==============================] - 0s 11ms/step - loss: 0.1534 - accuracy: 0.9476 - val_loss: 0.3726 - val_accuracy: 0.8585
model.evaluate(x_test, y_test)    
#782/782 [==============================] - 3s 3ms/step - loss: 0.3924 - accuracy: 0.8486
#[0.3924349546432495, 0.8485600352287292]


# fewer hidden units per layer
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(8, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(8, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
#Epoch 10/10
#30/30 [==============================] - 0s 11ms/step - loss: 0.0804 - accuracy: 0.9785 - val_loss: 0.3270 - val_accuracy: 0.8796
model.evaluate(x_test, y_test)    
#782/782 [==============================] - 3s 3ms/step - loss: 0.3580 - accuracy: 0.8693
#[0.3579915761947632, 0.8692800402641296]


# use mse as loss
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.binary_accuracy])
model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
#Epoch 10/10
#30/30 [==============================] - 0s 11ms/step - loss: 0.0168 - accuracy: 0.9857 - val_loss: 0.0929 - val_accuracy: 0.8770
model.evaluate(x_test, y_test)
#782/782 [==============================] - 3s 4ms/step - loss: 0.1006 - accuracy: 0.8658
#[0.10062553733587265, 0.8658000826835632]


# use tanh as activation
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(16, activation='tanh', input_shape=(10000,)))
model.add(tf.keras.layers.Dense(16, activation='tanh'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=[tf.keras.metrics.binary_accuracy])
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=['acc'])
model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=512, validation_data=(x_val, y_val))
#Epoch 10/10
#30/30 [==============================] - 0s 10ms/step - loss: 0.0113 - binary_accuracy: 0.9877 - val_loss: 0.1063 - val_binary_accuracy: 0.8738
model.evaluate(x_test, y_test)
#782/782 [==============================] - 3s 3ms/step - loss: 0.1169 - binary_accuracy: 0.8592
#[0.11692535132169724, 0.8592400550842285]