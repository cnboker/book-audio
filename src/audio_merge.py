from pydub import AudioSegment
from glob import glob

files = sorted(glob("audio/part_*.wav"))
final = AudioSegment.empty()

for f in files:
    final += AudioSegment.from_wav(f) + AudioSegment.silent(300)  # 句子之间留 0.3 秒

final.export("book_final.mp3", format="mp3", bitrate="192k")
