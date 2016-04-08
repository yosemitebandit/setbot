"""Train a simple deep CNN on the CIFAR10 small images dataset.
"""

import json
import os

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np


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

# Output params.
architecture_filepath = '/tmp/setbot-cnn-model-architecture.json'
weights_filepath = '/tmp/setbot-cnn-weights.h5'
loss_history_filepath = '/tmp/setbot-cnn-loss-history.h5'

# Setup data and labels.
# todo: shuffle?
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
split_index = int(test_proportion * number_of_samples)
X_test = X[0:split_index]
y_test = y[0:split_index]
X_train = X[split_index:]
y_train = y[split_index:]
print 'X_train shape:', X_train.shape
print X_train.shape[0], 'train samples'
print X_test.shape[0], 'test samples'

# Convert class vectors to binary class matrices.
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Prepare to save architecture, weights and loss history.
model_architecture = model.to_json()
with open(architecture_filepath, 'w') as model_file:
  model_file.write(model_architecture)
base_weights_filepath = weights_filepath.split('.')[0]
weights_filepath = '%s.{epoch:02d}-{val_loss:.2f}.h5' % base_weights_filepath
weight_saver = ModelCheckpoint(
  filepath=weights_filepath, verbose=1, save_best_only=True)

class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(logs.get('loss'))

  def on_epoch_end(self, epoch, logs={}):
    print 'saving loss history in "%s"' % loss_history_filepath
    loss_history_data = json.dumps(self.losses)
    with open(loss_history_filepath) as loss_history_file:
      loss_history_file.write(loss_history_data)

history_saver = LossHistory()


# Train.
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, validation_data=(X_test, Y_test), shuffle=True,
          callbacks=[weight_saver, history_saver])
