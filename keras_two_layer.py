"""Trains a two layer deep neural network."""

import json
import os

from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np


# Label params.
filenames = [f.split('.')[0] for f in os.listdir('card-images')]
filenames.sort()
labels = {}
for index, filename in enumerate(filenames):
  labels[filename] = index

# Training params.
batch_size = 32
nb_classes = 81
epochs = 200
validation_proportion = 0.1

# Image params.
card_width = 100
aspect_ratio = 0.64
card_height = int(card_width / aspect_ratio)  # 156
image_rows, image_cols = card_height, card_width
image_channels = 3

# Output params.
model_output_dir = '/var/models-for-setbot/keras-two-layer'
base_architecture_filename = 'two-layer-architecture'
base_weights_filename = 'two-layer-weights'
loss_history_output_dir = '/tmp'
base_loss_history_filename = 'two-layer-loss-history'

# Setup training / validation data splits.
input_directory = 'rgba-data'
all_filenames = os.listdir(input_directory)
np.random.shuffle(all_filenames)
split_index = int(validation_proportion * len(all_filenames))
validation_filenames = all_filenames[0:split_index]
training_filenames = all_filenames[split_index:]

# Setup generator.
def data_and_label_generator(filenames, samples):
  sample_limit = len(filenames)
  total_samples_processed = 0
  while True:
    if total_samples_processed + samples >= sample_limit:
      total_samples_processed = 0
    X = np.zeros((samples, image_channels*image_rows*image_cols))
    y = np.zeros((samples,))
    filenames_in_batch = filenames[
      total_samples_processed:total_samples_processed+samples]
    for index, filename in enumerate(filenames_in_batch):
      path = os.path.join(input_directory, filename)
      X[index, :] = np.load(path).flatten()
      characteristics = filename.split('-')[0:4]
      simple_name = '-'.join(characteristics)
      y[index] = labels[simple_name]
    total_samples_processed += samples
    X = X.astype('float32')
    X /= 255
    y = np_utils.to_categorical(y, nb_classes)
    yield (X, y)

# Build the model.
model = Sequential()
model.add(Dense(64, input_dim=image_channels*image_rows*image_cols))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

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


samples = 100
training_generator = data_and_label_generator(training_filenames, samples)
validation_generator = data_and_label_generator(validation_filenames, samples)
model.fit_generator(
  generator=training_generator,
  samples_per_epoch=len(training_filenames),
  nb_epoch=epochs,
  verbose=1,
  show_accuracy=True,
  validation_data=validation_generator,
  nb_val_samples=len(validation_filenames),
  callbacks=[weight_saver, history_saver],
)
