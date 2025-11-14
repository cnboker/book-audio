#conda activate deepseek-py310
#
from transformers import AutoModel, AutoTokenizer
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = 'deepseek-ai/DeepSeek-OCR'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, _attn_implementation='eager', trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
prompt = "<image>\nParse the pcb figure. "
prompt = "<pdf>\npdf OCR. "
image_file = '388089-1.pdf'
output_path = 'output'

res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, 
                  base_size = 1024, image_size = 1024, crop_mode=False, save_results = True, test_compress = True)