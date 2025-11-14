# 文件名: voice_clone.py
from openvoice import clone_voice
import os

def voice_clone():
    # 保证输出目录存在
    os.makedirs("audio_cloned", exist_ok=True)

    # 替换为你识别到的文本切片数量（如果你已经有 chunks 列表，就保留原写法）
    from glob import glob
    files = sorted(glob("audio/part_*.wav"))

    reference_voice = "wav/girl.wav"  # 你的参考声音文件（要模仿的声音）

    for i, src in enumerate(files):
        clone_voice(
            source=src,
            reference=reference_voice,
            output=f"audio_cloned/part_{i:05d}.wav"
        )

    print("✅ 换声完成，输出路径：audio_cloned/")
