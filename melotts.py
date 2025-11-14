from melo.api import TTS
from scipy.io.wavfile import write
import numpy as np

# 初始化中文模型
model = TTS(language='ZH', device='cuda')

sr = model.hps.data.sampling_rate
speaker_id = model.hps.data.spk2id["ZH"]  # 使用官方中文说话人

def tts_to_file(text, output_path, pause_sec=0.6):
    model.tts_to_file(
            text=text,
            speaker_id=speaker_id,
            speed=0.9,          # 语速可调
            output_path=output_path,
            
        )
       

