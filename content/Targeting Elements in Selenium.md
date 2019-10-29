Title: Targeting Elements in Selenium
Date: 10-28-2019
Slug: blog-4
cover: https://images.unsplash.com/photo-1432821596592-e2c18b78144f?ixlib=rb-1.2.1&auto=format&fit=crop&w=1050&q=80
Disclaimer: This was started as a blog outlining the process of web scraping with selenium. 4 days later, my frustration with selenium and majority of my attempts not working has made me decide to write about my struggles instead.

In recognition of the 39k people using this browser automation tool, I will be working with this tool until I have resolved my conflicts, or lost all my hair - whichever happens first.

For the purposes of not being vague, I think what I struggled most trying to spin up a simple example of going through multiple nested pages to extract information is things not working. Like the lamb I am, I have resorted to the internet every time selenium has hit me with errors (with web drivers, web drivers not being in path, multiple packages not existing, the web driver simply not working) and the internet has not been  able to resolve my conflicts very easily. But even all of that does not start to compare to the frustration that has come out of me trying to target elements in selenium. At multilple stages of working through this blog I have thought of ways I can simply extract href information through beatifulsoup and used that to scrape a nested page instead of having to target the element in selenium to .click()

But in this house we do not accept defeat so here I am with my blog about how to target elements in selenium.


## Install Selenium and webdrivers


Installing selenium is fairly simple:


```python
# ! pip install selenium
```

To get Selenium working, you will need a webdriver. Chromedriver for chrome, geckodriver for firefox. The webdriver file needs to be in the PATH. After a few hours spent on [stackoverflow](https://stackoverflow.com/questions/29858752/error-message-chromedriver-executable-needs-to-be-available-in-the-path), I achieved this through placing the file in the bin inside anacoda folder.


```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
```


```python
URL = 'https://www.seleniumhq.org/'
driver = webdriver.Firefox()
driver.implicitly_wait(30)
driver.get(URL)
```

Taking a quick look inside dir(driver), we find a few options to target elements.


```python
dir(driver)[58:76]
```




    ['find_element',
     'find_element_by_class_name',
     'find_element_by_css_selector',
     'find_element_by_id',
     'find_element_by_link_text',
     'find_element_by_name',
     'find_element_by_partial_link_text',
     'find_element_by_tag_name',
     'find_element_by_xpath',
     'find_elements',
     'find_elements_by_class_name',
     'find_elements_by_css_selector',
     'find_elements_by_id',
     'find_elements_by_link_text',
     'find_elements_by_name',
     'find_elements_by_partial_link_text',
     'find_elements_by_tag_name',
     'find_elements_by_xpath']



Notice that for every method listed here, there's a elements version of it that targets multiple items on the page.

## find_element_by_class_name

Now that we have spun up a browser, let's try to find the download button for selenium. Through a quick inspection, we see that the download button can be targeted through the div with a class name of "downloadBox"


```python
driver.find_element_by_class_name('downloadBox').click()
```

.click() clicks the button and brings us to the download page. Multiple targets can be found by using find_elements_by_class_name.

## find_element_by_css_selector

I found this useful to target something like a table, since css selectors such as p tags, a tags and divs are far too general to target specific elements.


```python
[i.text for i in driver.find_elements_by_css_selector('tr')][1:6]
```




    ['Java 3.141.59 2018-11-14 Download   Change log   Javadoc',
     'C# 3.14.0 2018-08-02 Download Change log API docs',
     'Ruby 3.14.0 2018-08-03 Download Change log API docs',
     'Python 3.14.0 2018-08-02 Download Change log API docs',
     'Javascript (Node) 4.0.0-alpha.1 2018-01-13 Download Change log API docs']



## find_element_by_class_name

I find that targeting an element by id is the safest way, since no two elements in a page will have the same id. The downside of it, though is that we cannot target multiple elements in the page thorugh one id.


```python
driver.find_element_by_id('menu_documentation').click()
```

## find_element_by_xpath

I was expecting id or classname to be the easiest way to target something on the page, but it turned out to be through xpath. Partly because how easy it is to get the xpath of an element.

![](https://i.imgur.com/8GvUXku.jpg)


```python
x_path = '/html/body/div[3]/div[2]/div[2]/div[2]/ul/li[3]/input'
driver.find_element_by_xpath(x_path).click()
```

## find_element_by_name

Items inside a form have names and can be targeted using their names. In this case, we will be targetting the search input at the top of the page.


```python
driver.find_element_by_name('q')
```




    <selenium.webdriver.firefox.webelement.FirefoxWebElement (session="aafb89e2-4968-4778-9939-310512d75018", element="e493bb0b-06e1-4b45-b850-1b860d09fbd7")>



Perhaps one of seleniums strengths lie in automated form completeion, which I will explore in another blog when I manage to be more comfortable with selenium.

The only use case of find_element I have found seem to be paired with By, which in turn achieves the same results as the mentioned methods.Targeting through tag name, link text and partial link text also exists. In most cases the functionality of targeting thorugh a tag name should be achievable thorugh css selectors. link text and partial link text has thie own usage but id, class_name and css selector targetting covered the broad share of my needs. Xpath, to me is the last resort for items that are notoriously hard to separate from the rest of the page.
