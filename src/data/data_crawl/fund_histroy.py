from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import pandas as pd
import requests_cache
from pandas import DataFrame
from PIL import Image
from io import BytesIO
import base64
import numpy as np

# 初始化缓存
requests_cache.install_cache('demo_cache', expire_after=36000)

options = Options()
options.add_argument('user-data-dir=C:\\Users\\20586\\AppData\\Local\\Google\\Chrome\\User Data')
options.add_argument('profile-directory=Default')
service = Service('chromedriver.exe')

driver = webdriver.Chrome(options=options)

#%%

driver.get('https://www.morningstar.cn/quickrank/default.aspx')

#%%

def get_gif_as_base64(img_element):
    script = """
    function getGifAsBase64(imgElement) {
        var canvas = document.createElement('canvas');
        var ctx = canvas.getContext('2d');
        canvas.width = imgElement.width;
        canvas.height = imgElement.height;
        ctx.drawImage(imgElement, 0, 0, imgElement.width, imgElement.height);
        return canvas.toDataURL('image/gif').split(',')[1];
    }

    var imgElement = arguments[0];
    return getGifAsBase64(imgElement);
    """
    return driver.execute_script(script, img_element)

def parse_table(driver) -> DataFrame:
    page_source = driver.page_source
    table = BeautifulSoup(page_source, 'html.parser').find_all('table')[-1]
    rows = []
    for row in table.find_all('tr'):
        cols = []
        for col in row.find_all(['td', 'th']):
            img_tag = col.find('img')
            if img_tag and img_tag.get('src').endswith('.gif'):
                base64_gif = get_gif_as_base64(driver.find_element_by_xpath(col.img['xpath']))
                cols.append(base64_gif)
            else:
                cols.append(col.get_text().strip())
        rows.append(cols)

    df = DataFrame(rows[1:], columns=rows[0])
    return df[['代码', '基金名称', '基金分类', '晨星评级(三年)', '晨星评级(五年)', '净值日期', '单位净值(元)', '净值日变动(元)', '今年以来回报(%)']]

df_origin = parse_table(driver)

#%%

def count_star(base64_string: str) -> int:
    # Convert base64 string to byte stream
    byte_stream = base64.b64decode(base64_string)
    image = Image.open(BytesIO(byte_stream))

    # Convert image to RGB array
    rgb_array = np.array(image)

    # Load pattern images and compute their hashes
    pattern_hashes = {}
    for i in range(6):
        pattern_image = Image.open(f'star/{i}.gif')
        pattern_hash = hash(pattern_image.tobytes())
        pattern_hashes[pattern_hash] = i

    # Compute the hash of the input image
    image_hash = hash(image.tobytes())

    # Return the corresponding pattern index or -1 if not found
    return pattern_hashes.get(image_hash, -1)

df = df_origin.copy()
df['晨星评级(三年)'] = df['晨星评级(三年)'].map(count_star)
df['晨星评级(五年)'] = df['晨星评级(五年)'].map(count_star)

df_star_counted = df

df_star_counted.to_csv('fund_list_1_300.csv', index=False, encoding='utf-8-sig', mode='w', header=True)

#%%

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

wait = WebDriverWait(driver, 30)

def save_to_csv(df: DataFrame, path: str):
    df.to_csv(path, index=False, encoding='utf-8-sig', mode='a')

while True:
    next_page_btn = wait.until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ctl00_cphMain_AspNetPager1"]/a[12]')))
    next_page_btn.click()
    df_tmp = parse_table(driver)
    df_tmp['晨星评级(三年)'] = df_tmp['晨星评级(三年)'].apply(count_star)
    df_tmp['晨星评级(五年)'] = df_tmp['晨星评级(五年)'].apply(count_star)
    df_tmp.to_csv('fund_list_1_300.csv', index=False, encoding='utf-8-sig', mode='a', header=False)

#%%
