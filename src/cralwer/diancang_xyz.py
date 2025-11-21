import time
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

options = webdriver.ChromeOptions()
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

driver.get("https://www.diancang.xyz/lishizhuanji/13549/")

# 执行 JS 获取 href 列表
js = """
return Array.from(
    document.querySelectorAll("#booklist > li > a[href]")
).map(x => x.href);
"""

href_list = driver.execute_script(js)

print(href_list)  # 得到 Python 字符串数组
# 保存目录
SAVE_DIR = "assets/book/上下五千年"

# 确保目录存在
os.makedirs(SAVE_DIR, exist_ok=True)

#booklist
# ----------------------------
# 主循环
# ----------------------------
START_INDEX = 224   # 从 224 号开始保存 0224.txt

for real_index, url in enumerate(href_list, start=1):

    if real_index < START_INDEX:
        continue  # 跳过前面的 1~223

    print(f"正在访问 {real_index}: {url}")

    try:
        driver.get(url)
        time.sleep(1)

        js = """
        return document.querySelector(
            "body > div:nth-child(4) > div.col-xs-12.col-sm-12.col-md-12.col-lg-12 > div > div"
        )?.innerText || "";
        """

        content = driver.execute_script(js)

        # 四位数字文件名
        filename = f"{real_index:04d}.txt"
        filepath = os.path.join(SAVE_DIR, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"已保存：{filepath}")

    except Exception as e:
        print(f"⚠️ 访问 {url} 时发生错误：{e}")
        print("跳过该条，继续下一条……")
        continue

    time.sleep(1)


driver.quit()
print("全部完成！")
