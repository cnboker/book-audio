from src.my_cosyvoice import cos_voice
from glob import glob

def batch_audio():
    files = sorted(glob('assets/ocr_output/page_*.md'))    
    for f in files:
        print(f'{f} begin')
        cos_voice(f)
        

if __name__ == '__main__' :
    batch_audio()
