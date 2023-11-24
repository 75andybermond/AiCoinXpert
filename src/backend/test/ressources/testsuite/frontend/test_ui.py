import time

from selenium import webdriver

options = webdriver.ChromeOptions()
options.add_argument("--ignore-ssl-errors=yes")
options.add_argument("--ignore-certificate-errors")
driver = webdriver.Remote(
    command_executor="http://localhost:4444/wd/hub", options=options
)

# navigate to browserstack.com
driver.get("http://localhost:8080")
driver.find_element_by_link_text("Testsigma Cloud").click()
driver.close()
driver.quit()
