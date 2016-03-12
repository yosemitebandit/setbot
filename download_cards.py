import os
import urllib
import urlparse

from selenium import webdriver


base_url = 'http://localhost:8000/svg-cards'
params = {
  'number': 2,
  'color': 'purple',
  'texture': 'solid',
  'shape': 'diamond',
}
url_params = urllib.urlencode(params)
url = '%s?%s' % (base_url, url_params)

driver = webdriver.PhantomJS(service_log_path='/dev/null')
driver.set_window_size(800, 650)
driver.get(url)
driver.save_screenshot('screenshot.png')

# The driver.close and .quit methods aren't working..
os.system('pkill phantomjs')
