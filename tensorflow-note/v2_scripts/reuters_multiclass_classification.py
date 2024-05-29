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
NUM_WORDS = 10000
# Retrieve the training sequences.
(train_data, train_labels), (test_data, test_labels) = tf.keras.datasets.reuters.load_data(start_char=start_char, oov_char=oov_char, index_from=index_from, num_words=NUM_WORDS)
# Retrieve the word index file mapping words to indices
word_index = tf.keras.datasets.reuters.get_word_index()
# Reverse the word index to obtain a dict mapping indices to words
# And add `index_from` to indices to sync with `x_train`
inverted_word_index = dict((i + index_from, word) for (word, i) in word_index.items())
# Update `inverted_word_index` to include `start_char` and `oov_char`
inverted_word_index[start_char] = "[START]"
inverted_word_index[oov_char] = "[OOV]"
# Decode the first sequence in the dataset
decoded_sequence = " ".join(inverted_word_index[i] for i in train_data[0])


def vectorize_sequences(seqs, dims=NUM_WORDS):
 res = np.zeros((len(seqs), dims))
 for i, s in enumerate(seqs):
  res[i, s] = 1.
 return res

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# use built-in utilities
one_hot_train_labels = tf.keras.utils.to_categorical(train_labels)
one_hot_test_labels = tf.keras.utils.to_categorical(test_labels)

VALIDATION_DATA_SIZE=1000
x_val = x_train[:VALIDATION_DATA_SIZE]
partial_x_train = x_train[VALIDATION_DATA_SIZE:]
y_val = one_hot_train_labels[:VALIDATION_DATA_SIZE]
partial_y_train = one_hot_train_labels[VALIDATION_DATA_SIZE:] 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
#model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#Epoch 20/20
#16/16 [==============================] - 0s 19ms/step - loss: 0.1099 - accuracy: 0.9605 - val_loss: 1.0879 - val_accuracy: 0.8030

history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
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
plt.savefig('{}/dev-repo/tf-scaffold/images/reuters/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
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
plt.savefig('{}/dev-repo/tf-scaffold/images/reuters/img_{}.jpg'.format(os.getenv('HOME'), int(time.time())))
plt.show()


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.categorical_crossentropy, metrics=['acc'])
history = model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val))
#Epoch 9/9
#16/16 [==============================] - 0s 11ms/step - loss: 0.2729 - acc: 0.9399 - val_loss: 0.9079 - val_acc: 0.8150
model.evaluate(x_test, one_hot_test_labels)
#71/71 [==============================] - 1s 8ms/step - loss: 1.0060 - acc: 0.7823
#[1.0059919357299805, 0.7822796106338501]
predictions = model.predict(x_test)
predictions.shape
#(2246, 46)
np.sum(predictions[0])
#1.0
np.argmax(predictions[0])


# baseline
import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array))/len(test_labels)
#0.21193232413178986


y_train = np.array(train_labels)
y_test = np.array(test_labels)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
y_val = y_train[:VALIDATION_DATA_SIZE]
partial_y_train = y_train[VALIDATION_DATA_SIZE:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#Epoch 20/20
#16/16 [==============================] - 0s 8ms/step - loss: 0.1164 - acc: 0.9560 - val_loss: 1.0463 - val_acc: 0.8090
model.evaluate(x_test, y_test)
#71/71 [==============================] - 0s 4ms/step - loss: 1.1924 - acc: 0.7898
#[1.192440152168274, 0.7898486256599426]


# fewer hidden neurons
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))
#Epoch 20/20
#63/63 [==============================] - 0s 7ms/step - loss: 0.6619 - acc: 0.8120 - val_loss: 2.0763 - val_acc: 0.6740
model.evaluate(x_test, y_test)
#71/71 [==============================] - 0s 4ms/step - loss: 2.3504 - acc: 0.6416
#[2.350416898727417, 0.6415850520133972]


# more hidden neurons
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#Epoch 20/20
#16/16 [==============================] - 0s 9ms/step - loss: 0.1065 - acc: 0.9563 - val_loss: 1.1270 - val_acc: 0.7930
model.evaluate(x_test, y_test)
#71/71 [==============================] - 0s 4ms/step - loss: 1.3230 - acc: 0.7778
#[1.322984218597412, 0.777827262878418]


# fewer hidden layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#Epoch 20/20
#16/16 [==============================] - 0s 8ms/step - loss: 0.1077 - acc: 0.9582 - val_loss: 0.9486 - val_acc: 0.8140
model.evaluate(x_test, y_test)
#71/71 [==============================] - 0s 4ms/step - loss: 1.0748 - acc: 0.7925
#[1.074783205986023, 0.7925200462341309]


# more hidden layers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(64, activation='relu', input_shape=(NUM_WORDS,)))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(46, activation='softmax'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['acc'])
model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
#Epoch 20/20
#16/16 [==============================] - 0s 9ms/step - loss: 0.1261 - acc: 0.9575 - val_loss: 1.2160 - val_acc: 0.7970
model.evaluate(x_test, y_test)
#71/71 [==============================] - 0s 4ms/step - loss: 1.4196 - acc: 0.7747
#[1.4196401834487915, 0.7747105956077576]
