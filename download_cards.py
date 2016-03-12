"""Downloads images of all cards.
"""

import os
import urllib

from selenium import webdriver


# Setup the webdriver and output path.
driver = webdriver.PhantomJS(service_log_path='/dev/null')
driver.set_window_size(800, 650)
base_url = 'http://localhost:8000/svg-cards'
out_path = 'card-images'
if not os.path.exists(out_path):
  os.makedirs(out_path)

# List all valid params.
numbers = (1, 2, 3)
colors = ('red', 'green', 'purple')
textures = ('solid', 'empty', 'striped')
shapes = ('oval', 'diamond', 'bean')

# Capture an image for each permutation.
for number in numbers:
  for color in colors:
    for texture in textures:
      for shape in shapes:
        params = {
          'number': number,
          'color': color,
          'texture': texture,
          'shape': shape,
        }
        url_params = urllib.urlencode(params)
        url = '%s?%s' % (base_url, url_params)
        driver.get(url)
        filename = '%s-%s-%s-%s.png' % (number, color, texture, shape)
        print filename
        save_path = os.path.join(out_path, filename)
        driver.save_screenshot(save_path)

# The driver.close and .quit methods aren't working..
os.system('pkill phantomjs')
