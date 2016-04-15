"""Testing the keras model.

Usage:
  test_keras_model.py <image-filepath>

Arguments:
  <image-filepath>  path to the scaled image
"""

import os

from docopt import docopt
from keras.models import model_from_json
from PIL import Image
import numpy as np


# Card params.
channels = 3
aspect_ratio = 0.64
width = 100
height = int(width / aspect_ratio)
image_rows, image_cols = height, width

# Get args.
args = docopt(__doc__)

# Load the model.
architecture_path = '/var/models-for-setbot/updated-keras-cnn-with-generator/architecture.json'
weights_path = '/var/models-for-setbot/updated-keras-cnn-with-generator/weights.020-0.05.h5'
print 'loading model..'
with open(architecture_path) as architecture_file:
  model = model_from_json(architecture_file.read())
model.load_weights(weights_path)
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
)
print 'done.'

# Setup labels.
filename_labels = [f.split('.')[0] for f in os.listdir('card-images')]
filename_labels.sort()

# Load the image and massage it into the input array.
X = np.zeros((1, channels, image_rows, image_cols))
data = np.array(Image.open(args['<image-filepath>']))
data = data.reshape(3, image_rows, image_cols)
#data = np.load(args['<image-filepath>'])
X[0, :, :, :] = data
X = X.astype('float32')
X /= 255

# Predict.
prediction = model.predict_proba(X, verbose=0)[0]
probabilities = []
for index, probability in enumerate(prediction):
  if probability > 0.01:
    probabilities.append((filename_labels[index], probability))
probabilities = sorted(probabilities, key=lambda p: p[1], reverse=True)

# Print.
for name, probability in probabilities:
  print '%s -> %d%%' % (name, 100*probability)
