import os
import sys
import time
import wave
import pyaudio
from vosk import Model, KaldiRecognizer, SetLogLevel

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_vosk_realtime():
    """测试Vosk实时语音识别"""
    print("开始Vosk实时语音识别测试...")
    
    # 设置Vosk模型路径
    model_path = "vosk-model-cn-0.22"
    if not os.path.exists(model_path):
        print(f"错误：找不到语音模型 {model_path}")
        return
    
    # 加载模型
    model = Model(model_path)
    
    # 设置录音参数
    CHUNK = 4096
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    
    # 初始化PyAudio
    p = pyaudio.PyAudio()
    
    try:
        # 打开音频流
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        # 创建识别器
        rec = KaldiRecognizer(model, RATE)
        
        print("\n开始录音（按Ctrl+C停止）...")
        print("请对着麦克风说话...\n")
        
        while True:
            # 读取音频数据
            data = stream.read(CHUNK, exception_on_overflow=False)
            
            # 识别音频
            if rec.AcceptWaveform(data):
                result = rec.Result()
                if result:
                    print("识别结果:", result)
                    
    except KeyboardInterrupt:
        print("\n录音已停止")
    finally:
        # 关闭资源
        stream.stop_stream()
        stream.close()
        p.terminate()

def test_vosk_from_file(audio_file: str):
    """测试从WAV文件进行语音识别"""
    print(f"\n开始从文件识别测试: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"错误：音频文件不存在 {audio_file}")
        return
    
    # 加载模型
    model_path = "vosk-model-cn-0.22"
    if not os.path.exists(model_path):
        print(f"错误：找不到语音模型 {model_path}")
        return
    
    model = Model(model_path)
    
    try:
        # 打开WAV文件
        with wave.open(audio_file, "rb") as wf:
            # 验证音频格式
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
                print(f"错误：音频必须是16kHz、16bit、单声道WAV格式")
                print(f"当前格式：声道数={wf.getnchannels()}, "
                      f"采样位数={wf.getsampwidth()}, "
                      f"采样率={wf.getframerate()}")
                return
            
            # 创建识别器
            rec = KaldiRecognizer(model, wf.getframerate())
            rec.SetWords(True)
            
            # 分块读取并识别
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = rec.Result()
                    print("中间结果:", result)
            
            # 获取最终结果
            final_result = rec.FinalResult()
            print("\n最终识别结果:", final_result)
            
    except Exception as e:
        print(f"识别过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

def record_test_audio(duration: int = 5, output_file: str = "test_audio.wav"):
    """录制测试音频文件"""
    print(f"\n开始录制测试音频 (持续{duration}秒)...")
    
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # 确保采样率为16kHz
    
    p = pyaudio.PyAudio()
    
    try:
        # 打开音频流
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)
        
        print("请对着麦克风说话...")
        
        # 录制音频
        frames = []
        for i in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
            progress = (i + 1) / int(RATE / CHUNK * duration) * 100
            print(f"\r录音进度: {progress:.1f}%", end="")
        
        print("\n录音完成")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存为WAV文件
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)  # 16位采样，固定为2字节
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            
        print(f"音频已保存到: {output_file}")
        
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    # 测试实时语音识别
    print("=== 测试1：实时语音识别 ===")
    test_vosk_realtime()
    
    # 录制测试音频
    print("\n=== 测试2：录制测试音频 ===")
    record_test_audio(5, "temp/test_audio.wav")
    
    # 从文件识别
    print("\n=== 测试3：从文件识别 ===")
    test_vosk_from_file("temp/test_audio.wav")
