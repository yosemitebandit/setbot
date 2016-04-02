"""Webcam test.

via http://stackoverflow.com/a/19084592/232638
"""

import time
import cv2


cv2.namedWindow('preview')
vc = cv2.VideoCapture(0)
rval, frame = vc.read()

while True:
  # Show in preview window.
  if frame is not None:
     cv2.imshow('preview', frame)

  # Save.
  cv2.imwrite('/tmp/out.png', frame)

  # Capture another.
  rval, frame = vc.read()

  # Break on 'q'.
  if cv2.waitKey(1) & 0xFF == ord('q'):
     break

  # Wait.
  time.sleep(1)
