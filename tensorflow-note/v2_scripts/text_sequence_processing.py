#!/usr/bin/env python3

import os, time, json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#fp = '{}/dev-repo/tf-scaffold/images/imdb/img_{}.jpg'.format(os.getenv('HOME'), int(time.time()))


def word_level_one_hot_encoding_example():
 samples = ['The cat sat on the mat.', 'The dog ate my homework.']
 token_index = {}
 for s in samples:
  for w in s.split():
   if w not in token_index:
    token_index[w] = len(token_index)+1
 token_index
 #{'The': 1, 'cat': 2, 'sat': 3, 'on': 4, 'the': 5, 'mat.': 6, 'dog': 7, 'ate': 8, 'my': 9, 'homework.': 10}
 max_length = 10
 results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))
 for i, s in enumerate(samples):
  for j, w in list(enumerate(s.split()))[:max_length]:
   index = token_index.get(w)
   results[i, j, index] = 1.
 results.shape
 #(2, 10, 11)
 # batch_index, example_index, feature_index


def character_level_one_hot_encoding_example():
 import string
 chars = string.printable
 token_index = dict(zip(chars, range(1, len(chars)+1)))
 token_index
 samples = ['The cat sat on the mat.', 'The dog ate my homework.']
 max_length = 50
 results = np.zeros(shape=(len(samples), max_length, max(token_index.values())+1))
 for i, s in enumerate(samples):
  for j, w in enumerate(s[:max_length]):
   index = token_index.get(w)
   results[i, j, index] = 1.


# word-level encoding with keras utilities
'''
samples= ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(samples) # compact encoding
sequences = tokenizer.texts_to_sequences(samples) # compact encoding
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary') # ditto, except result is a matrix
#type(sequences)
#<class 'list'>
#sequences[0]
#[1, 2, 3, 4, 1, 5]
#word_index = tokenizer.word_index
#for k, v in word_index.items():
# print(k, v)
#the 1
#cat 2
#sat 3
#on 4
#mat 5
#dog 6
#ate 7
#my 8
#homework 9
#type(one_hot_results)
#<class 'numpy.ndarray'>
#one_hot_results.shape
#(2, 1000)
#one_hot_results[0]
'''


def word_level_one_hot_encoding_with_hash_example():
 samples = ['The cat sat on the mat.', 'The dog ate my homework.']
 max_dimension = 1000
 max_length = 10
 results = np.zeros(shape=(len(samples), max_length, max_dimension))
 for i, s in enumerate(samples):
  for j, w in list(enumerate(s.split()))[:max_length]:
   index = abs(hash(w))%max_dimension
   results[i, j, index] = 1.
 results.shape


# naive example to training a nn model on imdb dataset with word embedding 
max_features = 10000
max_len = 20
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=max_features)
x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(10000, 8, input_length=max_len))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.losses.binary_crossentropy, metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
#Epoch 10/10
#625/625 [==============================] - 4s 7ms/step - loss: 0.3021 - acc: 0.8754 - val_loss: 0.5268 - val_acc: 0.7548


url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
dataset = tf.keras.utils.get_file("aclImdb_v1", url, untar=True)
dataset_dir = os.path.join(os.getenv('HOME'), '.keras/datasets')
imdb_dir = os.path.join(dataset_dir, 'aclImdb')
#os.listdir(imdb_dir)
#['imdbEr.txt', 'test', 'imdb.vocab', 'README', 'train']
train_dir = os.path.join(imdb_dir, 'train')
labels = []
texts = []
for label_type in ['neg', 'pos']:
 dir_name = os.path.join(train_dir, label_type)
 for fname in os.listdir(dir_name):
  if fname.endswith('.txt'):
   with open(os.path.join(dir_name, fname)) as f:
    texts.append(f.read())
   labels.append(0 if label_type=='neg' else 1)

maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
#type(sequences)
#<class 'list'>
#len(sequences)
#25000
#word_index = tokenizer.word_index
#len(word_index)
#88582
data = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
#data.shape
#(25000, 100)
#labels.shape
#(25000,)
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples+validation_samples]
y_val = labels[training_samples:training_samples+validation_samples]


glove_dir = os.path.join(dataset_dir, 'glove')
embedding_index = {}
with open(os.path.join(glove_dir, 'glove.6B.100d.txt')) as f:
 for line in f:
  values = line.split()
  word = values[0]
  coefs = np.asarray(values[1:], dtype='float32')
  embedding_index[word] = coefs

#len(embedding_index)
#400000

embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
 if i < max_words:
  embedding_vector = embedding_index.get(word, None)
  if embedding_vector is not None:
   embedding_matrix[i] = embedding_vector

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
#model.summary()
# prohibit backpropagation from updating weights of embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
#Epoch 10/10
#7/7 [==============================] - 1s 152ms/step - loss: 0.1121 - acc: 0.9750 - val_loss: 0.8025 - val_acc: 0.5652
model.save_weights('pre_trained_glove_model.h5')

# more trainning data on model with pre-trained embedding
#
# training_samples = 10000, validation_samples = 2000
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 3s 3ms/step - loss: 1.1588 - acc: 0.6587
# [1.1587802171707153, 0.6586800217628479]
#
# training_samples = 20000, validation_samples = 5000
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 3s 3ms/step - loss: 0.9374 - acc: 0.7079
# [0.9374181628227234, 0.7078800201416016]

#history.history.keys()
#dict_keys(['loss', 'acc', 'val_loss', 'val_acc'])

def plot_acc_and_loss():
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

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss=tf.keras.losses.binary_crossentropy, metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
#Epoch 10/10
#7/7 [==============================] - 1s 168ms/step - loss: 0.0046 - acc: 1.0000 - val_loss: 0.7446 - val_acc: 0.5228

# evalute model performance on test_data
test_dir = os.path.join(imdb_dir, 'test')
labels = []
texts = []
for label_type in ['neg', 'pos']:
 dir_name = os.path.join(test_dir, label_type)
 for fname in os.listdir(dir_name):
  if fname.endswith('.txt'):
   with open(os.path.join(dir_name, fname)) as f:
    texts.append(f.read())
   labels.append(0 if label_type=='neg' else 1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

# evalute performance of model with pre-trained embedding on test_data
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)
#[0.7940724492073059, 0.5628000497817993]

# more trainning data on model with task-specific embedding
#
# training_samples = 10000, validation_samples = 2000
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 3s 3ms/step - loss: 1.0630 - acc: 0.8154
# [1.0630238056182861, 0.8154000639915466]
#
# training_samples = 20000, validation_samples = 5000
# model.evaluate(x_test, y_test)
# 782/782 [==============================] - 3s 3ms/step - loss: 1.3139 - acc: 0.8239
# [1.3139362335205078, 0.8239200711250305]
