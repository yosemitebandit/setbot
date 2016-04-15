"""Trains a CNN using a data generator."""

import json
import os

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np


# Training params.
batch_size = 100
classes = len(os.listdir('card-images'))
epochs = 200
samples_per_epoch = 10000
validation_samples = 1000

# Image params.
card_width = 100
aspect_ratio = 0.64
card_height = int(card_width / aspect_ratio)
image_rows, image_cols = card_height, card_width
image_channels = 3

# Output file info.
model_output_dir = '/var/models-for-setbot/updated-keras-cnn-with-generator-more-rectangles'
base_architecture_filename = 'architecture'
base_weights_filename = 'weights'
loss_history_output_dir = '/tmp'
base_loss_history_filename = 'updated-cnn-with-generator-more-rectangles-loss-history'

# Setup labels.
label_names = [f.split('.')[0] for f in os.listdir('card-images')]
label_names.sort()
labels = {}
for index, name in enumerate(label_names):
  labels[name] = index

# Setup training and validation splits.
input_directory = 'rgba-data'
all_filenames = os.listdir(input_directory)
np.random.shuffle(all_filenames)
validation_filenames = all_filenames[0:validation_samples]
training_filenames = all_filenames[validation_samples:]

def batch_generator(filenames, output_samples):
  """Generates a batch of training data, X and y.

  Args:
    filenames: all the filenames from which to draw
    output_samples: the number of datapoints to yield each iteration
  """
  samples_processed = 0
  samples_available = len(filenames)
  np.random.shuffle(filenames)
  while True:
    if samples_processed + output_samples > samples_available:
      np.random.shuffle(filenames)
      samples_processed = 0

    X = np.zeros((output_samples, image_channels, image_rows, image_cols))
    y = np.zeros((output_samples,)).astype(int)

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

model.add(Convolution2D(32, 3, 3, border_mode='valid',
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

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(
  loss='categorical_crossentropy',
  optimizer=adam,
  metrics=['accuracy'],
)

# Save model architecture.
model_architecture = model.to_json()
architecture_filename = '%s.json' % base_architecture_filename
architecture_filepath = os.path.join(model_output_dir, architecture_filename)
with open(architecture_filepath, 'w') as model_file:
  model_file.write(model_architecture)

# Prepare to save weights.
weights_filename = '%s.{epoch:03d}-{val_loss:.2f}.h5' % base_weights_filename
weights_filepath = os.path.join(model_output_dir, weights_filename)
weight_saver = ModelCheckpoint(filepath=weights_filepath, verbose=1)

# Save loss every epoch.
class LossHistory(Callback):
  def on_train_begin(self, logs={}):
    self.losses = []

  def on_batch_end(self, batch, logs={}):
    self.losses.append(float(logs.get('loss')))

  def on_epoch_end(self, epoch, logs={}):
    loss_history_data = json.dumps(self.losses)
    loss_history_filename = '%s.epoch:%03d.json' % (
      base_loss_history_filename, epoch)
    loss_history_filepath = os.path.join(
      loss_history_output_dir, loss_history_filename)
    print 'saving loss history to "%s"' % loss_history_filepath
    with open(loss_history_filepath, 'w') as loss_history_file:
      loss_history_file.write(loss_history_data)

history_saver = LossHistory()


# Train.
if __name__ == '__main__':
  training_generator = batch_generator(training_filenames, batch_size)
  validation_generator = batch_generator(validation_filenames, batch_size)
  model.fit_generator(
    generator=training_generator,
    samples_per_epoch=samples_per_epoch,
    nb_epoch=epochs,
    verbose=1,
    callbacks=[weight_saver, history_saver],
    validation_data=validation_generator,
    nb_val_samples=validation_samples,
  )
