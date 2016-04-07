"""Card isolation test."""

import time
import cv2
import numpy as np


cv2.namedWindow('preview')
cv2.createTrackbar('sensitivity', 'preview', 150, 255, lambda x: x)

vc = cv2.VideoCapture(0)
rval, frame = vc.read()

area_difference_threshold = 0.2
max_number_of_cards = 18
cards_per_col = 3
channels = 3
aspect_ratio = 0.64
width = 150
height = int(width / aspect_ratio)
h = np.array([[0, height], [0, 0], [width, 0], [width, height]], np.float32)

while True:
  # Show in preview window.
  if frame is not None:

    # Set threshold from slider.
    sensitivity = cv2.getTrackbarPos('sensitivity', 'preview')
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])


    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_img, lower_white, upper_white)

    contours, _ = cv2.findContours(
      thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    first_six_contours = sorted(
      contours, key=cv2.contourArea, reverse=True)[0:6]
    areas = map(cv2.contourArea, first_six_contours)
    # We use median here rather than average because that seems to be more
    # resilient to encroaching hands.
    median_area = np.median(areas)

    possible_contours = sorted(
      contours, key=cv2.contourArea, reverse=True)[0:max_number_of_cards]
    contours = []
    for contour in possible_contours:
      diff = abs((cv2.contourArea(contour) - median_area) / median_area)
      if diff < area_difference_threshold:
        contours.append(contour)

    number_of_cards = len(contours)
    cards_per_row = number_of_cards / cards_per_col
    max_number_of_cols = max_number_of_cards / cards_per_col
    new_image = np.zeros(
      (height*cards_per_col, width*max_number_of_cols, channels), np.uint8)
    print 'sensitivity: %s, cards: %s' % (sensitivity, number_of_cards)

    #contours = sorted(
    #  contours, key=cv2.contourArea, reverse=True)[0:number_of_cards]

    # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

    # Sort contours by distance from top left.
    rectangles, nw_corners = [], []
    for contour in contours:
      rect = cv2.minAreaRect(contour)
      points = np.array(cv2.cv.BoxPoints(rect), np.float32)
      rectangles.append(points)

      # c1, c2 = tuple(points[0]), tuple(points[2])
      # cv2.rectangle(frame, c1, c2, (0, 255, 0), 3)

      # Find NW corner.
      west_points = sorted(points, key=lambda p: p[0])[:2]
      north_points = sorted(west_points, key=lambda p: p[1])
      nw_corners.append(north_points[0])

    nw_corners = sorted(
      nw_corners, key=lambda p: p[1], reverse=True)
    top_row = sorted(
      nw_corners[0:cards_per_row], key=lambda p: p[0], reverse=True)
    middle_row = sorted(nw_corners[cards_per_row:2*cards_per_row],
                        key=lambda p: p[0], reverse=True)
    bottom_row = sorted(nw_corners[2*cards_per_row:3*cards_per_row],
                        key=lambda p: p[0], reverse=True)
    top_row.extend(middle_row)
    top_row.extend(bottom_row)
    ordered_corners = top_row

    for index, corner in enumerate(ordered_corners):
      for points in rectangles:
        if corner not in points:
          continue
        transform = cv2.getPerspectiveTransform(points, h)
        warp = cv2.warpPerspective(frame, transform, (width, height))
        x_offset = height * (index / cards_per_row)
        y_offset = width * (index % cards_per_row)
        new_image[x_offset:x_offset + height, y_offset:y_offset + width,
                  :channels] = warp

    cv2.imshow('preview', new_image)


  # Save.
  cv2.imwrite('/tmp/out.png', new_image)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # Wait.
  time.sleep(0.1)
