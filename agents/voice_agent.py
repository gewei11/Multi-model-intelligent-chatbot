import os
import wave
import json
import numpy as np
import time
from typing import Dict, Any, Optional, Union, BinaryIO, List

from vosk import Model, KaldiRecognizer, SetLogLevel
from utils.logger import get_logger
from utils.helper_functions import retry, safe_json_loads
from tools.chattts_tools import ChatTTSUtil  # 修改导入路径

class VoiceAgent:
    """
    语音Agent，负责处理用户的语音输入，将语音转换为文本
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化语音Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("voice_agent")
        self.logger.info("语音Agent初始化")
        
        # 检查必要的依赖
        try:
            import pyaudio
            self.pyaudio = pyaudio
        except ImportError:
            self.logger.error("PyAudio未安装，录音功能将不可用。请运行：pip install pyaudio")
            self.pyaudio = None
        
        # 设置Vosk日志级别
        SetLogLevel(-1)
        
        # 录音相关参数
        self.chunk = 4096
        self.channels = 1
        self.sample_rate = config.get("voice", {}).get("sample_rate", 16000)
        
        # 初始化语音模型
        self.model_path = config.get("voice", {}).get("model_path", "vosk-model-cn-0.22")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"语音模型路径不存在: {self.model_path}")
        
        self.model = Model(self.model_path)
        self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
        self.recognizer.SetWords(True)
        
        # 初始化语音合成工具
        self.tts_util = ChatTTSUtil()  # 使用新的工具类

        self.logger.info("语音Agent初始化完成")
    
    def process(self, audio_data: Union[str, bytes, BinaryIO], context: Dict[str, Any] = None) -> str:
        """
        处理语音数据，转换为文本
        
        Args:
            audio_data: 语音数据，可以是文件路径、字节数据或文件对象
            context: 上下文信息
            
        Returns:
            识别出的文本
        """
        if context is None:
            context = {}
        
        self.logger.info("开始处理语音数据")
        
        try:
            # 根据输入类型处理语音数据
            if isinstance(audio_data, str):
                # 输入是文件路径
                text = self._recognize_from_file(audio_data)
            elif isinstance(audio_data, bytes):
                # 输入是字节数据
                text = self._recognize_from_bytes(audio_data)
            else:
                # 输入是文件对象
                text = self._recognize_from_file_object(audio_data)
            
            self.logger.info(f"语音识别结果: {text}")
            return text
        
        except Exception as e:
            self.logger.error(f"语音识别失败: {str(e)}")
            return "抱歉，语音识别失败，请重试或使用文字输入。"
    
    def _recognize_from_file(self, file_path: str) -> str:
        """
        从音频文件中识别文本
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            识别出的文本
        """
        self.logger.info(f"从文件识别: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"音频文件不存在: {file_path}")
        
        with wave.open(file_path, "rb") as wf:
            # 检查音频格式
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise ValueError("音频必须是16kHz、16bit、单声道WAV格式")
            
            # 创建识别器
            rec = KaldiRecognizer(self.model, wf.getframerate())
            rec.SetWords(True)
            
            # 逐块读取音频并识别
            results = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    results.append(result.get("text", ""))
            
            # 获取最终结果
            final_result = json.loads(rec.FinalResult())
            results.append(final_result.get("text", ""))
            
            # 合并结果
            text = " ".join(filter(None, results))
            return text
    
    def _recognize_from_bytes(self, audio_bytes: bytes) -> str:
        """
        从字节数据中识别文本
        
        Args:
            audio_bytes: 音频字节数据
            
        Returns:
            识别出的文本
        """
        self.logger.info("从字节数据识别")
        
        # 创建识别器
        rec = KaldiRecognizer(self.model, self.sample_rate)
        rec.SetWords(True)
        
        # 处理音频数据
        if rec.AcceptWaveform(audio_bytes):
            result = json.loads(rec.Result())
        else:
            result = json.loads(rec.FinalResult())
        
        return result.get("text", "")
    
    def _recognize_from_file_object(self, file_obj: BinaryIO) -> str:
        """
        从文件对象中识别文本
        
        Args:
            file_obj: 音频文件对象
            
        Returns:
            识别出的文本
        """
        self.logger.info("从文件对象识别")
        
        # 读取所有数据
        audio_bytes = file_obj.read()
        
        # 使用字节数据识别
        return self._recognize_from_bytes(audio_bytes)
    
    def record_audio(self, duration: int = 5, output_file: str = None) -> str:
        """
        录制音频并识别
        
        Args:
            duration: 录制时长（秒）
            output_file: 输出文件路径，如果为None则不保存
            
        Returns:
            识别出的文本
        """
        if not self.pyaudio:
            error_msg = ("录音功能需要安装 PyAudio 库。\n"
                         "Windows 用户请运行:\n"
                         "pip install pipwin\n"
                         "pipwin install pyaudio\n\n"
                         "其他系统用户请运行:\n"
                         "pip install pyaudio")
            self.logger.error(error_msg)
            return error_msg
            
        try:
            self.logger.info(f"开始录音，持续{duration}秒")
            
            # 设置录音参数
            CHUNK = 1024
            FORMAT = self.pyaudio.paInt16
            CHANNELS = 1
            RATE = self.sample_rate
            
            # 初始化PyAudio
            p = self.pyaudio.PyAudio()
            
            # 打开音频流
            stream = p.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            input=True,
                            frames_per_buffer=CHUNK)
            
            # 录制音频
            frames = []
            for i in range(0, int(RATE / CHUNK * duration)):
                data = stream.read(CHUNK)
                frames.append(data)
            
            # 停止录音
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            self.logger.info("录音完成")
            
            # 保存音频文件（如果需要）
            if output_file:
                with wave.open(output_file, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(p.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                self.logger.info(f"音频已保存到: {output_file}")
                
                # 从文件识别
                return self._recognize_from_file(output_file)
            else:
                # 直接从字节数据识别
                audio_data = b''.join(frames)
                return self._recognize_from_bytes(audio_data)
            
        except Exception as e:
            self.logger.error(f"录音失败: {str(e)}")
            return "录音失败，请检查麦克风设置"
    
    def start_recording(self) -> tuple:
        """开始录音"""
        if not self.pyaudio:
            self.logger.error("PyAudio未安装")
            return None, None

        try:
            # 初始化PyAudio
            p = self.pyaudio.PyAudio()
            
            # 打开录音流
            stream = p.open(format=self.pyaudio.paInt16,
                           channels=self.channels,
                           rate=self.sample_rate,
                           input=True,
                           frames_per_buffer=self.chunk)

            self.logger.info(f"开始录音... 采样率={self.sample_rate}Hz")
            return p, stream

        except Exception as e:
            self.logger.error(f"开始录音失败: {str(e)}")
            return None, None

    def stop_recording(self, p, stream, frames) -> str:
        """停止录音并识别"""
        try:
            if not frames:
                self.logger.error("没有录到音频数据")
                return ""

            # 停止录音
            stream.stop_stream()
            stream.close()
            p.terminate()

            # 保存录音数据
            audio_data = b''.join(frames)
            total_bytes = len(audio_data)
            self.logger.info(f"录音完成，总字节数: {total_bytes}")

            # 进行语音识别
            self.recognizer.AcceptWaveform(audio_data)
            result = json.loads(self.recognizer.FinalResult())
            text = result.get("text", "").strip()

            self.logger.info(f"识别结果: {text}")
            return text

        except Exception as e:
            self.logger.error(f"停止录音失败: {str(e)}")
            return ""

    def synthesize_speech(self, text: str) -> Optional[str]:
        """将文本转换为语音"""
        try:
            # 使用ChatTTS生成语音
            texts = [text]
            output_files = self.tts_util.generateSound(texts, savePath="temp/")
            
            if output_files:
                self.logger.info(f"语音合成成功: {output_files[0]}")
                return output_files[0]
            return None

        except Exception as e:
            self.logger.error(f"语音合成失败: {str(e)}")
            return None

    @retry(max_attempts=2)
    def transcribe(self, audio_file: str) -> str:
        """
        转写音频文件为文本（带重试机制）
        
        Args:
            audio_file: 音频文件路径
            
        Returns:
            识别出的文本
        """
        return self._recognize_from_file(audio_file)

    def process_input(self, audio_data: Union[str, bytes, BinaryIO, List], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理语音输入"""
        if context is None:
            context = {}

        try:
            # 处理音频列表数据
            if isinstance(audio_data, list):
                if not audio_data:
                    return {"error": "未收到音频数据"}

                # 将音频数据转换为WAV文件
                temp_file = os.path.join("temp", "recording.wav")
                os.makedirs("temp", exist_ok=True)

                # 保存为WAV文件
                with wave.open(temp_file, 'wb') as wf:
                    wf.setnchannels(1)  # 单声道
                    wf.setsampwidth(2)  # 16位采样
                    wf.setframerate(self.sample_rate)  # 16kHz采样率
                    wf.writeframes(b''.join(audio_data))

                self.logger.info(f"临时音频文件已保存: {temp_file}")
                
                try:
                    # 从WAV文件识别
                    recognized_text = self._recognize_from_file(temp_file)
                    self.logger.info(f"识别结果: {recognized_text}")

                    result = {
                        "recognized_text": recognized_text,
                        "sentiment_result": None
                    }

                    # 如果启用情感分析
                    if context.get("sentiment_enabled", True):
                        from tools.sentiment_tools import SentimentAnalysisTool
                        sentiment_tool = SentimentAnalysisTool(self.config)
                        result["sentiment_result"] = sentiment_tool.analyze_combined(recognized_text)

                    return result

                finally:
                    # 清理临时文件
                    try:
                        os.remove(temp_file)
                        self.logger.info("临时音频文件已清理")
                    except Exception as e:
                        self.logger.error(f"清理临时文件失败: {str(e)}")

            else:
                recognized_text = self.process(audio_data, context)
                return {
                    "recognized_text": recognized_text,
                    "sentiment_result": None
                }

        except Exception as e:
            self.logger.error(f"处理语音输入失败: {str(e)}")
            return {"error": f"处理失败: {str(e)}"}
