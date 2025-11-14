import cosyvoice

import soundfile as sf
import os
from src.sentences_chunk import sentence_splitting

tts = cosyvoice.load_model("cosyvoice-tts-zh")
os.makedirs("audio", exist_ok=True)


def cos_voice(file:str):
    chunks = sentence_splitting(file=file)
    for i, sentence in enumerate(chunks):
        audio = tts.tts(sentence)
        sf.write(f"audio/part_{i:05d}.wav", audio, 44100)
