#!/usr/bin/env python3
"""
fast_cosyvoice.py
极速版 CosyVoice 推理模板（针对 V100 优化）
注意：在生产环境先确保 ffmpeg 已安装，且 CUDA 驱动可用。
"""

import os
import sys
import re
import shutil
from pathlib import Path
from glob import glob
from typing import List

# 必须在 import cosyvoice 之前把源码路径加入
# 把 path 改成你本地 CosyVoice 源码位置
COSYVOICE_SRC = os.path.abspath("external/CosyVoice")
MATCHA_SRC = os.path.join(COSYVOICE_SRC, "third_party", "Matcha-TTS")
sys.path.insert(0, COSYVOICE_SRC)
sys.path.insert(0, MATCHA_SRC)

# 常用库
import torchaudio
from pydub import AudioSegment

# cosyvoice 导入
from cosyvoice.cli.cosyvoice import CosyVoice2
from cosyvoice.utils.file_utils import load_wav

# --------- 配置区（请按需修改） ---------
MODEL_DIR = os.path.abspath("external/CosyVoice/pretrained_models/CosyVoice2-0.5B")
PROMPT_WAV = os.path.abspath("external/CosyVoice/asset/zero_shot_prompt.wav")
AUDIO_DIR = os.path.abspath("assets/audio_parts")
AUDIO_TMP_DIR = os.path.abspath("audio")
OUT_DIR = AUDIO_DIR  # 合并输出放这里
SENTENCE_MAX_CHARS = 120     # 每段最大字符数（按中文字符）
BATCH_SENTENCES = 4          # 每次循环处理几句（不是内部并行，但有利于管理）
LOAD_JIT = False
LOAD_TRT = False   # 若你没有 TRT，init 会尝试回退
FP16 = True
# ----------------------------------------

# 全局变量（只加载一次）
cosyvoice = None
prompt_speech_16k = None

# ----------------- 辅助函数 -----------------
def check_ffmpeg():
    """验证 ffmpeg/ffprobe 存在"""
    def cmd_ok(cmd):
        try:
            import subprocess
            subprocess.run([cmd, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            return False
    if not cmd_ok("ffmpeg") or not cmd_ok("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe 未安装或不可执行。Ubuntu: sudo apt install ffmpeg")

def check_cuda():
    """简单检查 CUDA 可用性"""
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def split_sentences(text: str, max_chars: int = SENTENCE_MAX_CHARS) -> List[str]:
    """将长文本按标点拆成短句，并确保每句不超过 max_chars"""
    # 先按句号等拆
    parts = re.split(r'([。！？\.\!\?])', text)
    # 合并标点和句子
    sentences = []
    for i in range(0, len(parts)-1, 2):
        s = (parts[i] + parts[i+1]).strip()
        if s:
            sentences.append(s)
    # 可能尾部无标点
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1].strip())
    # 再把太长的句子切块
    out = []
    for s in sentences:
        if len(s) <= max_chars:
            out.append(s)
        else:
            # 简单分块
            for i in range(0, len(s), max_chars):
                out.append(s[i:i+max_chars])
    return out

def clear_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
        return
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isfile(fp) or os.path.islink(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception as e:
            print("clear_dir failed:", fp, e)

# -------------- 模型初始化 --------------
def init_model(model_dir=MODEL_DIR, prompt_wav=PROMPT_WAV, load_jit=LOAD_JIT, load_trt=LOAD_TRT, fp16=FP16):
    global cosyvoice, prompt_speech_16k
    if cosyvoice is not None and prompt_speech_16k is not None:
        return

    # 环境检测
    print("CUDA available:", check_cuda())
    try:
        check_ffmpeg()
    except RuntimeError as e:
        print("Warning:", e)
        # 不阻塞运行，但 pydub 会报错合并音频，建议先安装 ffmpeg

    # 尝试初始化（若 TRT 不可用则回退）
    tried_trt = load_trt
    while True:
        try:
            cosyvoice = CosyVoice2(model_dir, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
            break
        except Exception as e:
            print("Model init failed with load_trt=%s, load_jit=%s, fp16=%s -> %s" % (load_trt, load_jit, fp16, str(e)))
            if load_trt:
                print("Disabling TensorRT and retrying...")
                load_trt = False
                load_jit = False
                fp16 = True  # keep fp16 if GPU supports
                continue
            else:
                raise

    # 加载 prompt 音频
    prompt_speech_16k = load_wav(prompt_wav, 16000)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    print("Model initialized. sample_rate:", cosyvoice.sample_rate)

# -------------- 推理主逻辑 --------------
def cos_voice(md_file: str,output_dir:str):
    AUDIO_DIR = output_dir
    """
    将 md 文件文本分句并转换为单个 wav：
      - 若对应 wav 已存在则跳过
      - 结果保存在 AUDIO_DIR/<md_name>.wav
    """
    init_model()
    os.makedirs(AUDIO_DIR,exist_ok=True)
    wav_file = Path(md_file).stem + ".wav"
    wav_path = os.path.join(AUDIO_DIR, wav_file)
    if Path(wav_path).exists():
        print(f"file -> {wav_path} existed. skip.")
        return wav_path

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        print("empty text, skip")
        return None
    if len(text) < 30:
        print(f"text length {len(text)} < 30, skip")
        return None

    # 分句
    sentences = split_sentences(text)
    if not sentences:
        print("no sentences extracted, skip")
        return None

    # 清空临时分段音频目录
    clear_dir(AUDIO_TMP_DIR)

    # 按小批量处理（这里不是并行，但能保证模型复用）
    part_idx = 0
    for i in range(0, len(sentences), BATCH_SENTENCES):
        batch = sentences[i:i+BATCH_SENTENCES]
        # 将 batch 中每句分别推理（如果 cosyvoice 提供 batch api 可替换为 batch api）
        for sent in batch:
            print(f"gen part {part_idx}: {sent[:40]}...")
            # 这里使用你之前用的接口（stream=False）
            for k, out in enumerate(cosyvoice.inference_cross_lingual(sent, prompt_speech_16k, stream=False)):
                part_path = os.path.join(AUDIO_TMP_DIR, f"part_{part_idx:04d}.wav")
                torchaudio.save(part_path, out["tts_speech"], cosyvoice.sample_rate)
                part_idx += 1

    # 合并所有 part
    merged_files = sorted(glob(os.path.join(AUDIO_TMP_DIR, "part_*.wav")))
    if not merged_files:
        print("no parts generated")
        return None

    final = AudioSegment.empty()
    for f in merged_files:
        final += AudioSegment.from_wav(f) + AudioSegment.silent(300)  # 0.3s gap

    final.export(wav_path, format="wav", bitrate="192k")
    print("Saved merged wav:", wav_path)
    return wav_path

# -------------- CLI --------------
# if __name__ == "__main__":
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("mdfile", help="input md file path")
#     p.add_argument("--no-trt", action="store_true", help="disable TensorRT")
#     p.add_argument("--no-jit", action="store_true", help="disable JIT")
#     p.add_argument("--no-fp16", action="store_true", help="disable fp16")
#     args = p.parse_args()

#     if args.no_trt:
#         LOAD_TRT = False
#     if args.no_jit:
#         LOAD_JIT = False
#     if args.no_fp16:
#         FP16 = False

#     res = cos_voice(args.mdfile)
#     print("done ->", res)
