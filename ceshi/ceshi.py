import ChatTTS
import torch
import torchaudio

# 初始化 ChatTTS 对象
chat = ChatTTS.Chat()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
chat.load(source='huggingface', device=device)  # 设置为 True 以获得更好的性能

###################################
# Sample a speaker from Gaussian.

rand_spk = chat.sample_random_speaker()

params_infer_code = ChatTTS.Chat.InferCodeParams(
    spk_emb = rand_spk, # add sampled speaker 
    temperature = .3,   # using custom temperature
    top_P = 0.7,        # top P decode
    top_K = 20,         # top K decode
)

params_refine_text = ChatTTS.Chat.RefineTextParams(
    prompt='[oral_2][laugh_0][break_6]',
)

text = ["大家好，我是Chat T T S，欢迎来到畅的科技工坊。"]

wavs = chat.infer(
    text,
    params_refine_text=params_refine_text,
    params_infer_code=params_infer_code,
)

try:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
except:
    torchaudio.save("word_level_output.wav", torch.from_numpy(wavs[0]), 24000)