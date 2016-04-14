"""Trains a simple network with keras."""

import json
import os
import sys

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np


# Label params.
label_names = [f.split('.')[0] for f in os.listdir('card-images')]
label_names.sort()
labels = {}
for index, name in enumerate(label_names):
  labels[name] = index

# Image params.
card_width = 100
aspect_ratio = 0.64
card_height = int(card_width / aspect_ratio)  # 156
image_rows, image_cols = card_height, card_width
image_channels = 3

# Training params.
batch_size = 32
classes = len(label_names)
epochs = 200
validation_proportion = 0.2

# Setup training / validation data splits.
# input_directory = 'rgba-data'
# all_filenames = os.listdir(input_directory)
# np.random.shuffle(all_filenames)
# split_index = int(validation_proportion * len(all_filenames))
# validation_filenames = all_filenames[0:split_index]
# training_filenames = all_filenames[split_index:]
# print '%s training inputs, %s validation inputs' % (
  # len(training_filenames), len(validation_filenames))

# X_training = np.zeros((len(training_filenames), 3, image_rows, image_cols))
# y_training = np.zeros((len(training_filenames),))
# for index, filename in enumerate(training_filenames):
  # path = os.path.join(input_directory, filename)
  # X_training[index, :, :, :] = np.load(path)
  # simple_name = '-'.join(filename.split('-')[0:4])
  # y_training[index] = labels[simple_name]

# X_validation = np.zeros((len(validation_filenames), 3, image_rows, image_cols))
# y_validation = np.zeros((len(validation_filenames),))
# for index, filename in enumerate(validation_filenames):
  # path = os.path.join(input_directory, filename)
  # X_validation[index, :, :, :] = np.load(path)
  # simple_name = '-'.join(filename.split('-')[0:4])
  # y_validation[index] = labels[simple_name]

print 'loading data..'
input_directory = 'rgba-data'
number_of_samples = len(os.listdir(input_directory))
X = np.zeros((number_of_samples, image_channels, image_rows, image_cols))
y = np.zeros((number_of_samples,))
for index, filename in enumerate(os.listdir(input_directory)):
  path = os.path.join(input_directory, filename)
  X[index, :, :, :] = np.load(path)
  characteristics = filename.split('-')[0:4]
  simple_name = '-'.join(characteristics)
  y[index] = labels[simple_name]

# Divide into training / test splits.
split_index = int(validation_proportion * number_of_samples)
X_validation = X[0:split_index]
y_validation = y[0:split_index]
X_training = X[split_index:]
y_training = y[split_index:]
print 'X_train shape:', X_training.shape
print X_training.shape[0], 'train samples'
print X_validation.shape[0], 'test samples'

X_training = X_training.astype('float32')
X_validation = X_validation.astype('float32')
X_training /= 255
X_validation /= 255

Y_training = np_utils.to_categorical(y_training, classes)
Y_validation = np_utils.to_categorical(y_validation, classes)


# Build the model.
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(image_channels, image_rows, image_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(X_training, Y_training,
          batch_size=batch_size,
          nb_epoch=epochs,
          validation_data=(X_validation, Y_validation),
          show_accuracy=True,
          shuffle=True)

