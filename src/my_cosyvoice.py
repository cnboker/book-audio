from pathlib import Path
import sys
import os
import shutil
from pydub import AudioSegment
from glob import glob

# 必须在 import 之前
sys.path.append("external/CosyVoice")
sys.path.append('external/CosyVoice/third_party/Matcha-TTS')

from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
import torchaudio



cosyvoice = None
prompt_speech_16k = None
os.makedirs('audio',exist_ok=True)
os.makedirs("assets/audio_parts", exist_ok=True)

def init_model():
    global cosyvoice
    global prompt_speech_16k
    cosyvoice = CosyVoice2('external/CosyVoice/pretrained_models/CosyVoice2-0.5B', load_jit=False, load_trt=False, fp16=True)    
    prompt_speech_16k = load_wav('external/CosyVoice/asset/zero_shot_prompt.wav', 16000)

def cos_voice(file:str):
  
    #clear dir
    clear_dir('audio')
    #check file exists
    wav_file = Path(file).stem + '.wav'
    wav_path = os.path.join('assets/audio_parts', wav_file)

    # 检查完整路径
    if Path(wav_path).exists():
        print(f'file -> {wav_path} existed')
        return

    global cosyvoice
    global prompt_speech_16k

    if cos_voice is None or prompt_speech_16k is None:
        init_model()

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

  # 如果文本长度小于30，直接返回
    if text is None or len(text) < 30:
        print(f"文本长度 {len(text)} 小于30，跳过生成音频")
        return
    
    for i, j in enumerate(cosyvoice.inference_cross_lingual(text, prompt_speech_16k, stream=False,text_frontend=False )):
        torchaudio.save('audio/part_{}.wav'.format(i), j['tts_speech'], cosyvoice.sample_rate)
    out_file = Path(file).stem + '.wav'
    print('outfile->', out_file)
    audio_merge(f'assets/audio_parts/{out_file}')

def audio_merge(out_file:str):
    files = sorted(glob("audio/part_*.wav"))
    final = AudioSegment.empty()

    for f in files:
        final += AudioSegment.from_wav(f) + AudioSegment.silent(300)  # 句子之间留 0.3 秒

    final.export(out_file, format="wav", bitrate="192k")
    
def clear_dir(path):
    if not os.path.isdir(path):
        print(f"{path} is not a directory.")
        return

    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)          # 删除文件
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)      # 删除子目录
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

