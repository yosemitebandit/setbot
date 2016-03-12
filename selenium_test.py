import os

from selenium import webdriver


driver = webdriver.PhantomJS(service_log_path='/dev/null')
driver.set_window_size(800, 650)
driver.get('http://localhost:8000/svg-cards')
driver.save_screenshot('screenshot.png')

# The driver.close and .quit methods aren't working..
os.system('pkill phantomjs')


'''
# can't get this to work :/
body = driver.find_element_by_tag_name('body')
print body
body.send_keys('s')
'''
