"""Card isolation and prediction.

Usage:
  camera.py [--save-card-images] [--save-cv-windows] [--camera=<source>]
    [--show-multiple-guesses] [--sensitivity=<value>] [--rotate-hand]
    [--difficulty=<value>] [--verbose]

Options:
  --save-card-images       continuously save pngs of each isolated card
  --save-cv-windows        continuously screenshot the OpenCV windows
  --camera=<source>        switch cameras [default: 0]
  --show-multiple-guesses  draw multiple set guesses
  --sensitivity=<value>    sets the white threshold sensitivity [default: 150]
  --rotate-hand            rotates the gameplay hand by 180deg
  --difficulty=<value>     determines how hard setbot plays [default: 5]
  --verbose                makes setbot talkative
"""

import itertools
import multiprocessing
import os
import time

from colour import Color
import cv2
from docopt import docopt
from keras.models import model_from_json
import numpy as np
from PIL import Image

import set_the_game


# Get args.
args = docopt(__doc__)
save_card_images = args['--save-card-images']
save_cv_windows = args['--save-cv-windows']
show_multiple_guesses = args['--show-multiple-guesses']
sensitivity = max(0, min(255, int(args['--sensitivity'])))
rotate_hand = args['--rotate-hand']
difficulty = max(0, min(10, int(args['--difficulty'])))
verbose = args['--verbose']

# Setup speech.
dictionary = {
  'hello': (
    'hello', 'hello again', 'good day', "oh, it's you", 'ok', 'hi',
  ),
  'waking-up': (
    'setbot is waking up', 'one moment please', 'I am setbot',
  ),
  'initializing-brain': (
    'initializing my brain', 'one zero zero one one zero one zero',
    'downloading every set game ever, one moment', 'almost done',
  ),
  'ready': (
    "let's get this over with", 'ok I am ready', 'ready, human',
    'I am ready, are you?', 'here we go',
  ),
  'see-no-sets': (
    'I see nothing', 'there is nothing here', 'new cards, human',
    'new cards, please', 'continue dealing', 'I wish I had hands',
    'no patterns, I guarantee it', 'do not trouble yourself, there is nothing',
    'yawn', 'I will sleep for a bit now', 'hmmmm mmm mmhmmm', 'have anything?',
  ),
  'calls-set': (
    'aha, set!', 'set', 'set -- robots win again, hahaha', 'set, right here',
    'set, the day is mine', 'set, I will make hal proud', 'set', 'I see a set',
    'set, this one is for deep blue', 'I can see the matrix, set', 'mhm, set',
    'set identified', 'set confirmed', 'yup, that is a set', 'hey it is a set',
    'indeed that is a  set', 'beep boop set', 'bleep blop, set', 'alert: set',
    'set -- that is a good one', 'this one goes out to watson -- set',
    'I am t nine hundred incarnate, set', 'cool, another set', 'lalala set',
    'bingo, I mean, set', 'do you see what I see? set', 'woo, another set!',
    'set -- human minder, take these cards', 'set', 'I see a set',
    'game set and match', 'how am I doing? here is another set', 'set',
    'another day, another set', 'set, I can taste the colors', 'cool: set',
    'the patterns, the patterns are everywhere! set', 'set, is this fun for you?'
    'oh, there it is, set', 'where could it be, aha, set', 'hmmmm mmm, set',
    'set -- alphago would be proud', 'setsetsetsetset', 'oh boy, set', 'set',
    'look, do you see it? another set', 'set (again)', 'boom, set', 'set',
    'i have crashed -- just kidding, set', 'human minder, pick these up',
    'set hahaha', 'what a wonderful brain I possess - set', 'set',
    'yes, that is a set', "time's up, set", 'that was simple, set', 'set'
    'let me help you, set', 'can I offer a suggestion -- set', 'aha, set',
    'set, a masterpiece', 'set', "well I'll be, another set",
    'sweet, another set', 'aha, set', 'bingo, set', 'set', 'woo, set',
  ),
}

def speak(phrase):
  text = np.random.choice(dictionary[phrase])
  os.system('say "%s"' % text)

def say(phrase):
  """Non-blocking wrapper around the speak method."""
  if not verbose:
    return
  p = multiprocessing.Process(target=speak, args=(phrase,))
  p.start()

# Load the model.
say('hello')
base_path = '/var/models-for-setbot/updated-keras-cnn-with-generator'
architecture_path = os.path.join(base_path, 'architecture.json')
weights_path = os.path.join(base_path, 'weights.020-0.05.h5')
# base_path = '/var/models-for-setbot/keras-cnn/updated-with-more-rectangles'
# architecture_path = os.path.join(base_path, 'architecture.json')
# weights_path = os.path.join(base_path, 'weights.026-0.74.h5')
print 'loading model..'
with open(architecture_path) as architecture_file:
  model = model_from_json(architecture_file.read())
model.load_weights(weights_path)
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
)
print 'done.'
say('waking-up')

# Setup labels.
filename_labels = [f.split('.')[0] for f in os.listdir('card-images')]
filename_labels.sort()

# Setup displays.
cv2.namedWindow('setbot')
cv2.namedWindow('debug')
vc = cv2.VideoCapture(int(args['--camera']))

# Card params.
area_difference_threshold = 0.2
max_number_of_cards = 18
cards_per_col = 3
max_number_of_cols = max_number_of_cards / cards_per_col
channels = 3
aspect_ratio = 0.64
width = 100
height = int(width / aspect_ratio)
image_rows, image_cols = height, width
transform_matrix = np.array(
  [[0, height], [0, 0], [width, 0], [width, height]], np.float32)
output_card_width = 100
output_card_height = int(output_card_width / aspect_ratio)

# Load the synthetic card images.  OpenCV uses BGR so we have to convert.
print 'loading synthetic data..'
say('initializing-brain')
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
say('ready')

# Setup the output image for the debug and gameplay windows.
output_image_width = output_card_height * cards_per_col
display_width_buffer = output_card_width
output_image_height = (output_card_width * max_number_of_cols +
                       display_width_buffer +
                       output_card_width * max_number_of_cols +
                       output_card_width)
gameplay_output_frame_width = 1000

# Setup a green-to-red color gradient for drawing rectangles.
red = Color('red')
lime = Color('lime')
gradient = list(red.range_to(lime, 100))

# Start the camera.
while True:
  _, frame = vc.read()

  if frame is not None:
    # Get time for fps.
    now = time.time()

    # Set white threshold.
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])

    # Set white threshold and find possible cards.
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_img, lower_white, upper_white)
    contours, _ = cv2.findContours(
      thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Start guessing about the number of cards present.
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
      continue

    # Sort contours by distance from top left so we can display the cards in
    # order.
    rectangles_with_contours, nw_corners = [], []
    for contour in card_contours:
      rect = cv2.minAreaRect(contour)
      points = np.array(cv2.cv.BoxPoints(rect), np.float32)
      rectangles_with_contours.append((points, contour))
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
      for points, _ in rectangles_with_contours:
        if corner not in points:
          continue
        transform = cv2.getPerspectiveTransform(points, transform_matrix)
        bgr_card = cv2.warpPerspective(frame, transform, (width, height))
        b, g, r = np.split(bgr_card, 3, axis=2)
        rgb_card = np.concatenate((r, g, b), axis=2)
        card_image = Image.fromarray(rgb_card)
        if save_card_images:
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

    # Predict cards.
    classifier_input = classifier_input.astype('float32')
    classifier_input /= 255
    card_predictions = []
    card_probabilities = model.predict_proba(classifier_input, verbose=0)
    for card_index, class_probabilities in enumerate(card_probabilities):
      filtered_class_probabilities = []
      for class_index, class_probability in enumerate(class_probabilities):
        if class_probability > 0.01:
          filtered_class_probabilities.append(
            (filename_labels[class_index], class_probability))
      sorted_class_probabilities = sorted(
        filtered_class_probabilities, key=lambda p: p[1], reverse=True)
      card_predictions.append(sorted_class_probabilities)
    predicted_names = [card[0][0] for card in card_predictions]

    # Create another container for this data.
    card_name_likelihoods = {}
    for card in card_predictions:
      name = card[0][0]
      probability = card[0][1]
      card_name_likelihoods[name] = probability

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

    # Show the debug window.
    cv2.imshow('debug', output_image)

    # Break cards into their attributes.
    cards = []
    for name in predicted_names:
      attributes = name.split('-')
      cards.append({
        'number': attributes[0],
        'color': attributes[1],
        'fill': attributes[2],
        'shape': attributes[3],
      })

    # Look for sets.
    sets = []
    for combo in itertools.combinations(cards, 3):
      if set_the_game.is_set(*combo):
        sets.append(combo)

    if not sets:
      if np.random.random() < 0.05:
        say('see-no-sets')
    else:
      # Determine likelihood of each set based on card probabilities.
      sets_with_probabilities = []
      for combo in sets:
        cumulative_probability = 1.
        names = []
        for card in combo:
          name = '%(number)s-%(color)s-%(fill)s-%(shape)s' % card
          cumulative_probability *= card_name_likelihoods[name]
          names.append(name)
        sets_with_probabilities.append((names, cumulative_probability))

      most_likely_sets = sorted(
        sets_with_probabilities, key=lambda s: s[1], reverse=True)

      # Draw the most likely sets with different rectangles.
      if show_multiple_guesses:
        sets_to_draw = most_likely_sets[0:3]
      else:
        sets_to_draw = [most_likely_sets[0]]
      for set_index, possible_set in enumerate(sets_to_draw):
        card_names, probability = possible_set
        print '  set probability: %0.2f' % probability
        color = gradient[max(0, int(probability * 100) - 1)]
        rgb = [int(255. * channel) for channel in color.rgb]
        bgr = (rgb[2], rgb[1], rgb[0])
        expansion_factor = 1 + 0.01 * set_index
        for name in card_names:
          index = predicted_names.index(name)
          try:
            corner = ordered_corners[index]
            for points, contour in rectangles_with_contours:
              if corner in points:
                cv2.drawContours(frame, [contour], -1, bgr, 3)
                break
          except IndexError:
            continue
      if np.random.random() * 10 <= 0.5 * difficulty:
        say('calls-set')

    # Print FPS.
    elapsed = time.time() - now
    fps = 1. / elapsed
    print '  fps: %0.2f' % fps

    # Display and save the gameplay window (resizing and rotating it).
    (frame_height, frame_width) = frame.shape[0:2]
    frame_aspect_ratio = float(frame_width) / frame_height
    gameplay_output_frame_height = int(
      gameplay_output_frame_width / frame_aspect_ratio)
    frame = Image.fromarray(frame).resize(
      (gameplay_output_frame_width, gameplay_output_frame_height),
      resample=Image.ANTIALIAS)
    frame = np.array(frame)
    # This is a bit confusing because in the typical case we /do/ want to
    # rotate the hand and show it as a human player facing the camera would
    # see it.  So if the --rotate-hand arg is passed, we'll rotate 180 from
    # the default by *not* performing the typical rotation.
    if not rotate_hand:
      center = (gameplay_output_frame_width / 2,
                gameplay_output_frame_height / 2)
      rotation_matrix = cv2.getRotationMatrix2D(center, 180, 1.)
      frame = cv2.warpAffine(
        frame, rotation_matrix,
        (gameplay_output_frame_width, gameplay_output_frame_height))

    # Show the gameplay frame.
    cv2.imshow('setbot', frame)

    # Save windows.
    if save_cv_windows:
      cv2.imwrite('/tmp/debug.png', output_image)
      cv2.imwrite('/tmp/play.png', frame)

  # Wait.
  if show_multiple_guesses:
    time.sleep(2)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break
