import re


def sentence_splitting(file:str):

    with open(file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 按标点断句
    sentences = re.split(r'([。！？\?])', text)
    chunks = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]

    # 去掉过短/无意义行
    chunks = [c.strip() for c in chunks if len(c.strip()) > 5]
    return chunks