"""Trains a CNN using a data generator."""

import json
import os
import sys

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np


# Training params.
batch_size = 32
classes = len(os.listdir('card-images'))
epochs = 200
test_proportion = 0.1

# Image params.
card_width = 100
aspect_ratio = 0.64
card_height = int(card_width / aspect_ratio)  # 156
image_rows, image_cols = card_height, card_width
image_channels = 3

# Setup labels.
label_names = [f.split('.')[0] for f in os.listdir('card-images')]
label_names.sort()
labels = {}
for index, name in enumerate(label_names):
  labels[name] = index

# Setup training data.
input_directory = 'rgba-data'
all_filenames = os.listdir(input_directory)
np.random.shuffle(all_filenames)


def batch_generator(filenames, output_samples):
  """Generates a batch of training data, X and y.

  Args:
    filenames: all the filenames from which to draw
    output_samples: the number of datapoints to yield each iteration
  """
  samples_processed = 0
  samples_available = len(filenames)
  np.random.shuffle(filenames)
  X = np.zeros((output_samples, image_channels, image_rows, image_cols))
  y = np.zeros((output_samples,))
  while True:
    if samples_processed + output_samples > samples_available:
      print 'resetting and shuffling..'
      np.random.shuffle(filenames)
      samples_processed = 0

    filenames_to_process = filenames[
      samples_processed:samples_processed+output_samples]
    for index, filename in enumerate(filenames_to_process):
      X[index, :, :, :] = np.load(os.path.join(input_directory, filename))
      y[index] = labels['-'.join(filename.split('-')[0:4])]
    X = X.astype('float32')
    X /= 255
    y = np_utils.to_categorical(y, classes)

    samples_processed += output_samples
    yield X, y


# Build the model.
model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=(image_channels, image_rows, image_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# model.add(Convolution2D(64, 3, 3, border_mode='same'))
# model.add(Activation('relu'))
# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(classes))
model.add(Activation('softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(
  loss='categorical_crossentropy',
  optimizer=adam,
  metrics=['accuracy'],
)


class PrintBatchLogs(Callback):
  def on_batch_end(self, epoch, logs={}):
    print logs
print_batch_logs_callback = PrintBatchLogs()

# Train.
data_generator = batch_generator(all_filenames, batch_size)
model.fit_generator(
  generator=data_generator,
  samples_per_epoch=batch_size*100,
  nb_epoch=epochs,
  verbose=1,
  callbacks=[print_batch_logs_callback],
)
