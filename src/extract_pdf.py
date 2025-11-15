# extract_pdf_official_fixed.py
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from glob import glob
# ====================== 1. 环境设置 ======================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 本地模型路径
model_dir = r"/home/scott/models/models--deepseek-ai--DeepSeek-OCR"

# ====================== 2. 加载模型和 tokenizer ======================
tokenizer = None
model = None


def init_model():
    global tokenizer
    global model

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()
    print("Model loaded!")

# ====================== 3. PDF → 临时图片文件夹 ======================
def pdf_to_images_temp(pdf_path, temp_dir="temp_ocr_images", dpi=200):
    os.makedirs(temp_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        mat = fitz.Matrix(dpi/72, dpi/72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_data = pix.tobytes("png")
        img_path = os.path.join(temp_dir, f"page_{page_num+1:04d}.png")
        with open(img_path, "wb") as f:
            f.write(img_data)
        image_paths.append(img_path)
        print(f"Page {page_num+1} → {img_path}")
        
    doc.close()
    return image_paths


# ====================== 4. 调用官方 infer（每页一张图）=====================
def ocr_with_infer(image_path, output_dir="assets/ocr_output"):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    image_paths = sorted(glob(f"{image_path}/*.png"))
    for idx, img_path in enumerate(image_paths):
        print(f"\nOCR Page {idx+1}/{len(image_paths)}: {img_path}")
        try:
            result = model.infer(
                tokenizer=tokenizer,
                prompt="<image>\n<|grounding|>Convert the document to txt.",
                image_file=img_path,           # 必须是图片路径！
                output_path=output_dir,
                base_size=1024,
                image_size=1024,
                crop_mode=False,
                save_results=True,
                test_compress=True
            )
            OUTPUT_DIR = Path(output_dir)
            output_path = OUTPUT_DIR / f"result.mmd"
            page_text = output_path.read_text(encoding="utf-8").strip() if output_path.exists() else None
        
            # 同时保存单页
            if page_text is not None and len(page_text) > 30:
                file = os.path.join(output_dir, f"page_{idx+1:04d}.md")
                print('write file->', file)
                all_results.append(f"\n--- Page {idx+1} ---\n{page_text}\n")
                with open(file, "w", encoding="utf-8") as f:
                    f.write(page_text)

        except Exception as e:
            error_msg = f"[OCR FAILED: {e}]"
            all_results.append(f"\n--- Page {idx+1} ---\n{error_msg}\n")
            print(error_msg)

    # 合并所有页
    full_text = "\n".join(all_results)
    with open(os.path.join(output_dir, "FULL_BOOK.md"), "w", encoding="utf-8") as f:
        f.write(full_text)

    return full_text

# ====================== 5. 主流程 ======================
if __name__ == "__main__":
    pdf_path = r"assets/book/AI+新生：破解人机共存密码：人类最后一个大问题+(斯图尔特·罗素).pdf"

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF 不存在：{pdf_path}")
    if model is None or tokenizer is None:
        init_model()
    print("Step 1: Converting PDF to images...")
    
    image_path = "assets/temp_ocr_images"  # 仅测试第一页   
    full_text = ocr_with_infer(image_path, output_dir="assets/ocr_output")

    print(f"\nOCR 完成！")
