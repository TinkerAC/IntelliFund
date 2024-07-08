#%%
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import base64
from PIL import Image
from io import BytesIO
import numpy as np
import time
from io import StringIO

# 初始化缓存（如果需要）
# requests_cache.install_cache('demo_cache', expire_after=36000)

options = Options()
options.add_argument('user-data-dir=C:\\Users\\20586\\AppData\\Local\\Google\\Chrome\\User Data')
options.add_argument('profile-directory=Default')
service = Service('chromedriver.exe')

driver = webdriver.Chrome(options=options, service=service)
#%%

driver.get('https://www.morningstar.cn/quickrank/default.aspx')
wait = WebDriverWait(driver, 30)
driver.execute_script("""
window.getGifAsBase64 = function(targetSrc) {
  const images = document.getElementsByTagName('img');
  let targetImage = null;
  for (let img of images) {
    if (img.src === targetSrc) {
      targetImage = img;
      break;
    }
  }
  if (!targetImage) {
    return null;
  }
  const canvas = document.createElement('canvas');
  const context = canvas.getContext('2d');
  canvas.width = targetImage.width;
  canvas.height = targetImage.height;
  context.drawImage(targetImage, 0, 0);
  const dataUrl = canvas.toDataURL('image/gif');
  const base64String = dataUrl.split(',')[1];
  return base64String;
}
""")


# 定义获取 base64 编码的后端方法
def get_image_base64(img_src):
    driver.execute_script(f"getGifAsBase64('{img_src}')")

#%%
def readhtml(driver):
    return pd.read_html(StringIO(driver.page_source))[-1]


# 解析表格
def parse_table(driver) -> DataFrame:
    try:
        page_source = driver.page_source
        table = BeautifulSoup(page_source, 'html.parser').find_all('table')[-1]
        rows = []
        for row in table.find_all('tr'):
            cols = []
            for col in row.find_all(['td', 'th']):
                img_tag = col.find('img')
                if img_tag:
                    image_base64 = get_image_base64(img_tag["src"])
                    cols.append(image_base64)
                else:
                    cols.append(col.text.strip())
            rows.append(cols)

        df = DataFrame(rows[1:], columns=rows[0])
        return df[['代码', '基金名称', '基金分类', '晨星评级(三年)', '晨星评级(五年)', '净值日期', '单位净值(元)',
                   '净值日变动(元)', '今年以来回报(%)']]
    except Exception as e:
        print(f"Error parsing table: {e}")
        return pd.DataFrame()


df_origin = parse_table(driver)
df_origin
#%%

# 预加载 star 图片
pattern_hashes = {}
for i in range(6):
    pattern_image = Image.open(f'star/{i}.gif')
    pattern_hash = hash(pattern_image.tobytes())
    pattern_hashes[pattern_hash] = i


def count_star(image_base64: str) -> int:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        image_hash = hash(image.tobytes())
        return pattern_hashes.get(image_hash, -1)
    except Exception as e:
        print(f"Error counting star: {e}")
        return -1

#%%

df = df_origin.copy()
df['晨星评级(三年)'] = df['晨星评级(三年)'].apply(count_star)
df['晨星评级(五年)'] = df['晨星评级(五年)'].apply(count_star)

df_star_counted = df

df_star_counted.to_csv('fund_list_1_300.csv', index=False, encoding='utf-8-sig', mode='w', header=True)

#%%
def save_to_csv(df: DataFrame, path: str):
    df.to_csv(path, index=False, encoding='utf-8-sig', mode='a')


while True:
    try:
        next_page_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="ctl00_cphMain_AspNetPager1"]/a[12]')))
        next_page_btn.click()
        time.sleep(2)  # 等待页面加载

        df_tmp = parse_table(driver)
        if df_tmp.empty:
            break

        df_tmp['晨星评级(三年)'] = df_tmp['晨星评级(三年)'].apply(count_star)
        df_tmp['晨星评级(五年)'] = df_tmp['晨星评级(五年)'].apply(count_star)
        save_to_csv(df_tmp, 'fund_list_1_300.csv')
    except Exception as e:
        print(f"Error in pagination: {e}")
        break
