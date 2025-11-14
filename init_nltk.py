# init_nltk.py
import nltk
import ssl

# 解决 SSL 证书问题（如果你在国内或公司网络）
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# 下载必须的资源
resources = [
    'averaged_perceptron_tagger_eng',
    'punkt',                    # 分句用
    'cmudict',                  # 发音词典（g2p_en 用）
]

for res in resources:
    print(f"正在下载: {res} ...")
    nltk.download(res, quiet=False)

print("NLTK 资源下载完成！现在可以正常运行 MeloTTS 了。")