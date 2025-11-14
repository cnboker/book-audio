from cosyvoice import CosyVoice

model = CosyVoice("cosyvoice-tts")
audio = model.tts("你好，我是 CosyVoice 模型。")
model.save(audio, "demo.wav")
