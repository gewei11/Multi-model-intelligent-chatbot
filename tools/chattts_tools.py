import ChatTTS
import torch
import soundfile as sf
import numpy as np
import os
import torchaudio
from utils.logger import get_logger

class ChatTTSUtil:
    """语音合成工具"""
    def __init__(self):
        self.logger = get_logger("chatTTS_util")
        self.chat = ChatTTS.Chat()
        
        try:
            # 初始化设备
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.logger.info(f"Using device: {self.device}")
            
            # 初始化模型
            self.logger.info("开始加载模型...")
            self.chat.load(source='huggingface', device=self.device)
            self.logger.info("模型加载完成")

            # 采样随机说话人
            self.rand_spk = self.chat.sample_random_speaker()
            
            # 设置参数
            self.params_refine_text = ChatTTS.Chat.RefineTextParams(
                prompt='[oral_2][laugh_0][break_6]'
            )
            
            self.params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.rand_spk,
                temperature=0.3,
                top_P=0.7,
                top_K=20
            )

        except Exception as e:
            self.logger.error(f"模型初始化失败: {str(e)}")
            raise

    def setRefineTextConf(self, oralConf="[oral_0]", laughConf="[laugh_0]", breakConf="[break_0]"):
        """设置文本精炼配置"""
        self.params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt=f"{oralConf}{laughConf}{breakConf}"
        )

    def setInferCode(self, temperature=0.3, top_P=0.7, top_K=20, speed="[speed_5]"):
        """设置推理参数"""
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.rand_spk,
            temperature=temperature,
            top_P=top_P,
            top_K=top_K
        )

    def generateSound(self, texts, savePath="output/", filePrefix="output"):
        """生成音频文件"""
        os.makedirs(savePath, exist_ok=True)
        
        try:
            # 生成音频波形
            wavs = self.chat.infer(
                texts,
                params_refine_text=self.params_refine_text,
                params_infer_code=self.params_infer_code
            )
            
            # 保存音频文件
            wavFilePath = []
            for index, wave in enumerate(wavs):
                wav_path = f"{savePath}{filePrefix}_{index}.wav"
                
                try:
                    # 参考test_ChatTTS.py的保存方式，使用torchaudio保存
                    try:
                        # 确保音频数据是2D张量，torchaudio.save需要形状为[channels, samples]的张量
                        audio_tensor = torch.from_numpy(wave[0])
                        # 检查维度，如果是1D张量，则使用unsqueeze(0)添加通道维度
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        torchaudio.save(wav_path, audio_tensor, 24000)
                    except Exception as e:
                        self.logger.warning(f"第一次尝试保存音频失败: {str(e)}，尝试备选方法")
                        # 备选保存方法
                        audio_tensor = torch.from_numpy(wave[0])
                        # 无论如何都添加通道维度，确保是2D张量
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        torchaudio.save(wav_path, audio_tensor, 24000)
                    
                    wavFilePath.append(wav_path)
                    self.logger.info(f"已保存音频文件: {wav_path}")
                    
                except Exception as e:
                    self.logger.error(f"保存音频失败: {str(e)}")
                    continue
            
            return wavFilePath
            
        except Exception as e:
            self.logger.error(f"生成音频失败: {str(e)}")
            return []

    def save_audio(self, waveform, sample_rate, file_path):
        """
        保存音频文件，使用 Torchaudio 的后端调度方式
        """
        try:
            # 确保音频数据是2D张量，torchaudio.save需要形状为[channels, samples]的张量
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)
            # 检查维度，如果是1D张量，则使用unsqueeze(0)添加通道维度
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            torchaudio.save(file_path, waveform, sample_rate, backend="sox_io")
        except Exception as e:
            self.logger.error(f"保存音频失败: {str(e)}")
            raise

# 测试代码
if __name__ == "__main__":
    chUtil = ChatTTSUtil()
    texts = [
        "大家好，我是Chat T T S，欢迎来到畅的科技工坊。",
        "太棒了，我竟然是第一位嘉宾。"
    ]
    chUtil.setInferCode(0.8, 0.7, 20)
    chUtil.generateSound(texts)
