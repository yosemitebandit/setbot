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
width = 50
height = int(width / aspect_ratio)
image_rows, image_cols = height, width

# Get args.
args = docopt(__doc__)

# Load the model.
architecture_path = '/var/models-for-setbot/keras-cnn/architecture.json'
weights_path = '/var/models-for-setbot/keras-cnn/weights.hdf5'
print 'loading model..'
with open(architecture_path) as architecture_file:
  model = model_from_json(architecture_file.read())
model.load_weights(weights_path)
print 'done.'

# Setup labels.
filename_labels = [f.split('.')[0] for f in os.listdir('card-images')]
filename_labels.sort()

# Load the image and massage it into the input array.
#image = Image.open(args['<image-filepath>'])
X = np.zeros((1, channels, image_rows, image_cols))
#X[0, :, :, :] = np.transpose(np.array(image), (2, 0, 1))
X[0, :, :, :] = np.load(args['<image-filepath>'])

X = X.astype('float32')
X /= 255

# Predict.
prediction = model.predict_proba(X)[0]
for index, probability in enumerate(prediction):
  if probability > 0.01:
    print '%s -> %d%%' % (filename_labels[index], 100*probability)
