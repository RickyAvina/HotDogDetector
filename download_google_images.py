from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import sys
import os
import requests
import urllib3
from urllib3.exceptions import InsecureRequestWarning
import time
from tqdm import trange
import lxml

urllib3.disable_warnings(InsecureRequestWarning)


def download_google_staticimages(searchurl, name, dirs, chromedriver, headless=True):

    options = webdriver.ChromeOptions()
    options.add_argument('--no-sandbox')
    if headless:
        options.add_argument('--headless')

    try:
        browser = webdriver.Chrome(chromedriver, options=options)
    except Exception as e:
        print(f'No found chromedriver in this environment.')
        print(f'Install on your machine. exception: {e}')
        sys.exit()

    options.add_argument("window-size=1200x600")

    browser.set_window_size(1280, 1024)
    browser.get(searchurl)
    time.sleep(1)

    print(f'Getting you a lot of images. This may take a few moments...')

    element = browser.find_element_by_tag_name('body')

    # Scroll down
    #for i in range(30):
    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

    print(f'Reached end of page.')
    time.sleep(0.5)
    print(f'Retry')
    time.sleep(0.5)

    try:
        # Below is in japanese "show more result" sentences. Change this word to your lanaguage if you require.
        browser.find_element_by_xpath('//input[@value="Show more results"]').click()
    except:
        pass

    # Scroll down 2
    for i in range(50):
        element.send_keys(Keys.PAGE_DOWN)
        time.sleep(0.3)

    try:
        browser.find_element_by_id('smb').click()
        for i in range(50):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)
    except:
        for i in range(10):
            element.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.3)

    #elements = browser.find_elements_by_xpath('//div[@id="islrg"]')
    #page_source = elements[0].get_attribute('innerHTML')
    page_source = browser.page_source

    soup = BeautifulSoup(page_source, 'lxml')
    images = soup.find_all('img')

    urls = []
    for i in trange(len(images), desc='finding images'):
        try:
            url = images[i]['data-src']
            if not url.find('https://'):
                urls.append(url)
        except:
            try:
                url = images[i]['src']
                if not url.find('https://'):
                    urls.append(images[i]['src'])
            except Exception as e:
                print(f'No found image sources.')
                print(e)

    # count = 0
    # if urls:
    #     for url in urls:
    #         try:
    #             res = requests.get(url, verify=False, stream=True)
    #             rawdata = res.raw.read()
    #             with open(os.path.join(dirs, 'img_' + str(count) + '.jpg'), 'wb') as f:
    #                 f.write(rawdata)
    #                 count += 1
    #         except Exception as e:
    #             print('Failed to write rawdata.')
    #             print(e)

    count = 0
    if urls:
        for i in trange(len(urls), desc='saving images'):
            try:
                res = requests.get(urls[i], verify=False, stream=True)
                rawdata = res.raw.read()
                with open(os.path.join(dirs, name+'_img_' + str(count) + '.jpg'), 'wb') as f:
                    f.write(rawdata)
                    count += 1
            except Exception as e:
                print('Failed to write rawdata.')
                print(e)

    browser.close()
    return count

def download_multiple(arguments):
    search_words = arguments["search_words"]
    dir = arguments["dir"]
    max_count = arguments.get('max_count', 1000)
    headless = arguments.get('headless', True)
    chrome_driver = arguments['chrome_driver']      # '/usr/local/custom_bin/chromedriver'

    total_count = 0
    for search_word in search_words:
        search_url = 'https://www.google.com/search?q='+search_word+'&source=lnms&tbm=isch'
        dirname = dir+"/"+search_word+"/"
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        print("finding pics of " + search_word)
        t0 = time.time()
        count = download_google_staticimages(search_url, search_word, dirname, chrome_driver, headless)
        total_count += count
        t1 = time.time()
        print("finished finding pics of {} in {} seconds".format(search_word, str(t1-t0)))


    print(f'Download completed. [Successful count = {total_count}].')



def download_images(arguments):
    """
    :param arguments: {search_words: ["hot dog", "potato"],
                       dir: "data",
                       max_count: 1000
                       chrome_driver = '/usr/local/custom_bin/chromedriver'
                       headless: True
                      }
                      search_words, dir, chrome_driver are necessary
    :return: None
    """

    search_url = 'https://www.google.com/search?q='
    for word in arguments['search_words']:
        search_url += word + "+"
    search_url = search_url[:-1] + '&source=lnms&tbm=isch'

    dirs = arguments['dir']
    max_count = arguments.get('max_count', 1000)
    headless = arguments.get('headless', True)

    chrome_driver = arguments['chrome_driver']      # '/usr/local/custom_bin/chromedriver'
    #
    # if not os.path.exists(dirs):
    #     os.mkdir(dirs)

    t0 = time.time()
    count = download_google_staticimages(search_url, dirs, chrome_driver, headless)
    t1 = time.time()

    total_time = t1 - t0
    print(f'\n')
    print(f'Download completed. [Successful count = {count}].')
    print(f'Total time is {str(total_time)} seconds.')


if __name__ == '__main__':
    download_multiple({"search_words": ["plane", "dog", "frankfurter", "hot dog", "sausage", "burger"],
                     "dir": "data",
                     "max_count": 1000,
                     "chrome_driver": "/usr/local/custom_bin/chromedriver",
                     "headless": True})
