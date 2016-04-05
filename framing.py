"""Image framing test.

via http://stackoverflow.com/a/19084592/232638
"""

import time
import cv2
import numpy as np


cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)
rval, frame = vc.read()
size = 900

while True:
  # Show in preview window.
  if frame is not None:

    # Find contours.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (1, 1), 10)
    _, thresholded = cv2.threshold(blur, 118, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
      thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get bounding rectangle.
    rect = cv2.minAreaRect(contours[0])
    r = np.array(cv2.cv.BoxPoints(rect), np.float32)

    # Warp.
    h = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    transform = cv2.getPerspectiveTransform(r, h)
    warp = cv2.warpPerspective(frame, transform, (size, size))

    # Display
    cv2.imshow('preview', warp)



  # Save.
  cv2.imwrite('/tmp/out.png', frame)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

  # Wait.
  time.sleep(0.1)
