from src.cosvoice_tts import cos_voice
from glob import glob

def batch_audio(txt_src, output_dir):
    files = sorted(glob(f'{txt_src}/*.txt'))    
    for f in files:
        print(f'{f} begin')
        cos_voice(f,output_dir)
        

if __name__ == '__main__' :
    batch_audio('assets/book/上下五千年','assets/audio_parts/上下五千年')
