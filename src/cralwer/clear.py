# 你的 TXT 文件目录
import os
TARGET_DIR = "assets/book/上下五千年"
# 要替换为 "" 的内容
remove_texts = [
    "上一章",
    "返回目录",
    "下一章"
]

# 遍历所有文件
for root, dirs, files in os.walk(TARGET_DIR):
    for filename in files:
        if filename.lower().endswith(".txt"):
            filepath = os.path.join(root, filename)
            print(f"处理文件：{filepath}")

            # 读取内容
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            # 执行替换
            for text in remove_texts:
                content = content.replace(text, "")

            # 保存覆盖
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

print("全部 TXT 处理完成！")