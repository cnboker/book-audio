import torch
from TTS.api import TTS
from functools import partial
from pydub import AudioSegment


# 正确方式：用 partial 包装原函数
original_torch_load = torch.load
torch.load = partial(original_torch_load, weights_only=False)


def generate_clean(sentence, out_path):
    # 1. 文本处理
    sentence = sentence.strip()
    if not sentence.endswith(('。', '！', '？', '”', '’')):
        sentence += '。'
    
    # 2. 生成
    tts.tts_to_file(text=sentence, file_path=out_path)
    
    # 3. 强制截尾音
    audio = AudioSegment.from_wav(out_path)
    if len(audio) > 1500:  # >1.5s 才处理
        audio = audio[:-800]  # 切掉最后 0.8s
    audio.export(out_path, format="wav")

# 初始化 TTS
tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")
generate_clean("我可以帮你写一个 完整的 Windows + Coqui Tacotron2-DDC 中文测试脚本，一次性解决所有 UnpicklingError，让你直接运行生成 output.wav。", "output.wav")

