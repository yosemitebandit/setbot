"""Train a simple deep CNN on the CIFAR10 small images dataset.
"""

import os

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
from PIL import Image


# Seed RNG.
np.random.seed(2)

# Label params.
filenames = [f.split('.')[0] for f in os.listdir('card-images')]
filenames.sort()
labels = {}
for index, filename in enumerate(filenames):
  labels[filename] = index

# Training params.
batch_size = 32
nb_classes = 81
nb_epoch = 200
test_proportion = 0.1

# Image params.
card_width = 50
aspect_ratio = 0.64
card_height = int(card_width / aspect_ratio)  # 78
image_rows, image_cols = card_height, card_width
image_channels = 3

# Setup data and labels.
# todo: shuffle?
print 'loading data..'
input_directory = 'rgba-data'
number_of_samples = len(os.listdir(input_directory))
X = np.zeros((number_of_samples, image_rows * image_cols * image_channels))
y = np.zeros((number_of_samples,))
for index, filename in enumerate(os.listdir(input_directory)):
  path = os.path.join(input_directory, filename)
  X[index, :] = np.load(path)
  characteristics = filename.split('-')[0:4]
  simple_name = '-'.join(characteristics)
  y[index] = labels[simple_name]

# Divide into training / test splits.
split_index = int(test_proportion * number_of_samples)
X_train = X[0:split_index]
y_train = y[0:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
print 'X_train shape:', X_train.shape
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
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
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

model.fit(X_train, Y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, Y_test), shuffle=True)
