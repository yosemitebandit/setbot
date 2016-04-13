"""Card isolation and prediction.

Usage:
  camera.py <mode> [--save-card-images] [--save-cv-window] [--camera=<source>]

Arguments:
  <mode>  show the image 'frame' (for calibration) or try to 'predict'

Options:
  --save-card-images  continuously save pngs of each isolated card
  --save-cv-window    continuously screenshot the whole OpenCV window
  --camera=<source>   switch between webcam and external camera [default: 0]
"""

import itertools
import os
import time

import cv2
from docopt import docopt
from keras.models import model_from_json
import numpy as np
from PIL import Image

import set_the_game


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

# Setup display.
cv2.namedWindow('preview')
cv2.createTrackbar('sensitivity', 'preview', 150, 255, lambda x: x)
vc = cv2.VideoCapture(int(args['--camera']))

# Card params.
area_difference_threshold = 0.2
max_number_of_cards = 18
cards_per_col = 3
max_number_of_cols = max_number_of_cards / cards_per_col
channels = 3
aspect_ratio = 0.64
width = 50
height = int(width / aspect_ratio)
image_rows, image_cols = height, width
transform_matrix = np.array(
  [[0, height], [0, 0], [width, 0], [width, height]], np.float32)
output_card_width = 100
output_card_height = int(output_card_width / aspect_ratio)

# Load the synthetic card images.  OpenCV uses BGR so we have to convert.
print 'loading synthetic data..'
input_directory = 'card-images'
original_card_width, original_card_height = 352, 550
rendered_card_data = {}
for filename in os.listdir(input_directory):
  name = filename.split('.')[0]
  path = os.path.join(input_directory, filename)
  image = Image.open(path)
  r, g, b, _ = image.split()
  image = Image.merge('RGB', (b, g, r))
  image = image.crop((0, 0, original_card_width, original_card_height))
  image = image.resize((width, height), resample=Image.ANTIALIAS)
  rendered_card_data[name] = np.array(image)
print 'done.'

# Setup the output image for the OpenCV window.
output_image_width = output_card_height * cards_per_col
display_width_buffer = output_card_width
output_image_height = (output_card_width * max_number_of_cols +
                       display_width_buffer +
                       output_card_width * max_number_of_cols +
                       output_card_width)

# Start the camera.
while True:
  _, frame = vc.read()

  if frame is not None:
    # Get time for fps.
    now = time.time()

    # Set white threshold from slider.
    sensitivity = cv2.getTrackbarPos('sensitivity', 'preview')
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_img, lower_white, upper_white)

    contours, _ = cv2.findContours(
      thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Start guessing how many cards are visible.
    first_six_contours = sorted(
      contours, key=cv2.contourArea, reverse=True)[0:6]
    areas = map(cv2.contourArea, first_six_contours)
    # We use median here rather than average because that seems to be more
    # resilient to encroaching hands.
    median_area = np.median(areas)

    possible_contours = sorted(
      contours, key=cv2.contourArea, reverse=True)[0:max_number_of_cards]
    card_contours = []
    for contour in possible_contours:
      diff = abs((cv2.contourArea(contour) - median_area) / median_area)
      if diff < area_difference_threshold:
        card_contours.append(contour)

    number_of_cards = len(card_contours)
    cards_per_row = number_of_cards / cards_per_col
    if number_of_cards == 0 or cards_per_row == 0:
      print 'not enough cards in view..'
      time.sleep(0.2)
      continue

    # Sort contours by distance from top left so we can display the cards in
    # order.
    rectangles, nw_corners = [], []
    for contour in card_contours:
      rect = cv2.minAreaRect(contour)
      points = np.array(cv2.cv.BoxPoints(rect), np.float32)
      rectangles.append(points)
      # Find NW corner.
      west_points = sorted(points, key=lambda p: p[0])[:2]
      north_points = sorted(west_points, key=lambda p: p[1])
      nw_corners.append(north_points[0])

    nw_corners = sorted(nw_corners, key=lambda p: p[1], reverse=True)
    top_row = sorted(nw_corners[0:cards_per_row], key=lambda p: p[0],
                     reverse=True)
    middle_row = sorted(nw_corners[cards_per_row:2*cards_per_row],
                        key=lambda p: p[0], reverse=True)
    bottom_row = sorted(nw_corners[2*cards_per_row:3*cards_per_row],
                        key=lambda p: p[0], reverse=True)
    ordered_corners = []
    ordered_corners.extend(top_row)
    ordered_corners.extend(middle_row)
    ordered_corners.extend(bottom_row)

    # Draw the cards as the camera sees them and capture input data for the
    # classifier.
    output_image = np.zeros(
      (output_image_width, output_image_height, channels), np.uint8)
    classifier_input = np.zeros(
      (number_of_cards, channels, image_rows, image_cols))
    for index, corner in enumerate(ordered_corners):
      for points in rectangles:
        if corner not in points:
          continue
        transform = cv2.getPerspectiveTransform(points, transform_matrix)
        bgr_card = cv2.warpPerspective(frame, transform, (width, height))
        b, g, r = np.split(bgr_card, 3, axis=2)
        rgb_card = np.concatenate((r, g, b), axis=2)
        card_image = Image.fromarray(rgb_card)
        if args['--save-card-images']:
          filepath = '/tmp/%02d.png' % index
          card_image.save(filepath)
        data = rgb_card.reshape(3, image_rows, image_cols)
        classifier_input[index, :, :, :] = data
        output_card_data = Image.fromarray(bgr_card).resize(
          (output_card_width, output_card_height), resample=Image.ANTIALIAS)
        x_offset = output_card_height * (index / cards_per_row)
        y_offset = output_card_width * (index % cards_per_row)
        output_image[
          x_offset:x_offset + output_card_height,
          y_offset:y_offset + output_card_width,
          :channels
        ] = output_card_data

    classifier_input = classifier_input.astype('float32')
    classifier_input /= 255
    prediction = model.predict_classes(classifier_input, verbose=False)
    predicted_names = [filename_labels[i] for i in prediction]

    # Draw the estimate.
    for index, name in enumerate(predicted_names):
      data = Image.fromarray(rendered_card_data[name]).resize(
        (output_card_width, output_card_height), resample=Image.ANTIALIAS)
      x_offset = output_card_height * (index / cards_per_row)
      y_offset = (max_number_of_cols * output_card_width +
                  display_width_buffer +
                  output_card_width * (index % cards_per_row))
      try:
        output_image[
          x_offset:x_offset + output_card_height,
          y_offset:y_offset + output_card_width,
          :channels
        ] = data
      except ValueError:
        continue

    # Look for sets.
    cards = []
    for name in predicted_names:
      attributes = name.split('-')
      cards.append({
        'number': attributes[0],
        'color': attributes[1],
        'fill': attributes[2],
        'shape': attributes[3],
      })
    set_found = False
    for combo in itertools.combinations(cards, 3):
      if set_found:
        continue
      if set_the_game.is_set(*combo):
        for card in combo:
          name = '%(number)s-%(color)s-%(fill)s-%(shape)s' % card
          index = predicted_names.index(name)
          try:
            corner = ordered_corners[index]
            for points in rectangles:
              if corner not in points:
                continue
              c1, c2 = tuple(points[0]), tuple(points[2])
              cv2.rectangle(frame, c1, c2, (0, 255, 255), 3)
          except IndexError:
            continue
        set_found = True

    # Print FPS and print other params.
    elapsed = time.time() - now
    fps = 1. / elapsed
    print 'sensitivity: %s, cards: %s, fps: %0.2f' % (
      sensitivity, number_of_cards, fps)

    # Display and save.
    if args['<mode>'] == 'frame':
      #cv2.drawContours(frame, card_contours, -1, (0, 255, 0), 2)
      cv2.imshow('preview', frame)
      if args['--save-cv-window']:
        cv2.imwrite('/tmp/camera-frame.png', frame)
    elif args['<mode>'] == 'predict':
      cv2.imshow('preview', output_image)
      if args['--save-cv-window']:
        cv2.imwrite('/tmp/perception-and-prediction.png', output_image)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
