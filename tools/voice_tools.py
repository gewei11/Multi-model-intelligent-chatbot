import os
import wave
import time
import numpy as np
from typing import Dict, Any, Optional, Union
from vosk import Model, KaldiRecognizer, SetLogLevel
import soundfile as sf
import torch
import torchaudio
import ChatTTS
from dataclasses import dataclass

@dataclass
class VoiceRecognitionTool:
    """语音识别工具"""
    def __init__(self, config: Dict[str, Any]):
        self.model_path = config.get("voice", {}).get("model_path", "vosk-model-cn-0.22")
        self.sample_rate = config.get("voice", {}).get("sample_rate", 16000)
        
        # 初始化Vosk
        SetLogLevel(-1)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"语音模型路径不存在: {self.model_path}")
        self.model = Model(self.model_path)
    
    def recognize(self, audio_data: bytes) -> str:
        """语音识别"""
        try:
            # 创建临时WAV文件
            temp_file = "temp/temp_audio.wav"
            os.makedirs("temp", exist_ok=True)

            # 将音频数据写入WAV文件
            with wave.open(temp_file, 'wb') as wf:
                wf.setnchannels(1)  # 单声道
                wf.setsampwidth(2)  # 16位采样
                wf.setframerate(self.sample_rate)  # 采样率
                wf.writeframes(audio_data)

            # 从WAV文件读取数据进行识别
            with wave.open(temp_file, 'rb') as wf:
                recognizer = KaldiRecognizer(self.model, wf.getframerate())
                recognizer.SetWords(True)

                while True:
                    data = wf.readframes(4000)  # 每次读取4000帧
                    if len(data) == 0:
                        break
                    recognizer.AcceptWaveform(data)

                # 获取最终结果
                result = json.loads(recognizer.FinalResult())
                text = result.get("text", "").strip()

                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)

                return text if text else "未能识别语音内容"

        except Exception as e:
            print(f"语音识别错误: {str(e)}")
            return "语音识别失败，请重试"

@dataclass
class ChatTTSTool:
    """语音合成工具"""
    def __init__(self, config: Dict[str, Any]):
        self.output_dir = "output/voice"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 初始化ChatTTS
        self.chat = ChatTTS.Chat()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chat.load(source="huggingface", device=self.device)
        
        # 使用ChatTTS的标准参数类
        self.rand_spk = self.chat.sample_random_speaker()
        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]'
        )
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.rand_spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20
        )

    def synthesize(self, text: str) -> str:
        """语音合成"""
        output_path = os.path.join(self.output_dir, f"response_{int(time.time())}.wav")
        try:
            # 生成语音
            wavs = self.chat.infer(
                [text],
                params_refine_text=self.params_refine_text,
                params_infer_code=self.params_infer_code
            )
            
            # 使用 soundfile 保存音频
            audio_data = wavs[0][0]
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()
            sf.write(output_path, audio_data, 24000)
            return output_path
        except Exception as e:
            print(f"语音合成失败: {e}")
            return None
