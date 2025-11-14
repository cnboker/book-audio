import os
from voice_main import clean_md, tts_and_merge
md_path = "ocr_output/page_0006.md"
md_path="shi.txt"
OUTPUT_DIR = ""
with open(md_path, "r", encoding="utf-8") as f:
    raw = f.read()
text = clean_md(raw)

if not text:
    print("  警告：内容为空，跳过")
    exit()

# 输出 wav 路径（同名）
wav_name = os.path.splitext(os.path.basename(md_path))[0] + ".wav"
wav_path = os.path.join(OUTPUT_DIR, wav_name)

# TTS + 合并
tts_and_merge(text, wav_path)
