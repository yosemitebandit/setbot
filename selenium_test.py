from selenium import webdriver

driver = webdriver.PhantomJS(service_log_path='/dev/null')
driver.set_window_size(1024, 768)
driver.get('http://localhost:8000/svg-cards')
driver.save_screenshot('screenshot.png')

'''
# can't get this to work :/
body = driver.find_element_by_tag_name('body')
print body
body.send_keys('s')
'''
