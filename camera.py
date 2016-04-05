"""Card isolation test."""

import time
import cv2
import numpy as np


cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)
rval, frame = vc.read()
size = 900

sensitivity = 80
lower_white = np.array([0, 0, 255-sensitivity])
upper_white = np.array([255, sensitivity, 255])

while True:
  # Show in preview window.
  if frame is not None:

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    thresh = cv2.inRange(hsv_img, lower_white, upper_white)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[0:18]

    # Sort contours by distance from top left.
    rectangles, nw_corners = [], []
    for contour in contours:
      rect = cv2.minAreaRect(contour)
      points = np.array(cv2.cv.BoxPoints(rect), np.float32)
      rectangles.append(points)
      # Find NW corner.
      west_points = sorted(points, key=lambda p: p[0])[:2]
      north_points = sorted(west_points, key=lambda p: p[1])
      nw_corners.append(north_points[0])

    nw_corners = sorted(nw_corners, key=lambda p: p[1], reverse=True)
    top_row = sorted(nw_corners[0:6], key=lambda p: p[0], reverse=True)
    middle_row = sorted(nw_corners[6:12], key=lambda p: p[0], reverse=True)
    bottom_row = sorted(nw_corners[12:18], key=lambda p: p[0], reverse=True)

    top_row.extend(middle_row)
    top_row.extend(bottom_row)
    ordered_corners = top_row

    size = 150
    new_image = np.zeros((size, size, 3), np.float32)
    for index, corner in enumerate(ordered_corners):
      for points in rectangles:
        if corner not in points:
          continue
        h = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
        transform = cv2.getPerspectiveTransform(points, h)
        warp = cv2.warpPerspective(frame, transform, (size, size))
        x_offset = 150 * (index % 6)
        y_offset = 150 * (index / 6)
        #print x_offset, y_offset
        #new_image[x_offset:x_offset+size, y_offset:y_offset+size] = warp
        np.concatenate((new_image, warp), axis=0)

    cv2.imshow('preview', warp)


  # Save.
  #cv2.imwrite('/tmp/out.png', frame)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # Wait.
  time.sleep(0.1)
