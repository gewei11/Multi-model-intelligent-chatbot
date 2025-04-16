import os
import sys
import torch
import torchaudio
import ChatTTS
from typing import List

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_chattts():
    """测试ChatTTS功能"""
    print("开始ChatTTS测试...")
    
    # 初始化ChatTTS
    chat = ChatTTS.Chat()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载模型
        print("\n开始加载模型...")
        chat.load(source='huggingface', device=device)
        print("模型加载完成")
        
        # 采样说话人
        rand_spk = chat.sample_random_speaker()
        
        # 设置参数
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            temperature=0.3,
            top_P=0.7,
            top_K=20,
        )
        
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_2][laugh_0][break_6]',
        )
        
        # 测试文本
        test_texts = [
            "大家好，我是ChatTTS，欢迎来到AI助手。",
            "让我来为您朗读这段文字。",
            "希望您觉得我的声音听起来自然流畅。"
        ]
        
        print("\n开始生成语音...")
        # 生成音频
        output_dir = "temp/tts_output"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, text in enumerate(test_texts):
            wavs = chat.infer(
                [text],
                params_refine_text=params_refine_text,
                params_infer_code=params_infer_code,
            )
            
            # 保存音频文件
            output_file = f"{output_dir}/test_{i}.wav"
            try:
                torchaudio.save(output_file, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
            except:
                torchaudio.save(output_file, torch.from_numpy(wavs[0]), 24000)
                
            print(f"生成音频: {output_file}")
            if os.path.exists(output_file):
                size = os.path.getsize(output_file)
                print(f"文件大小: {size/1024:.1f}KB")
                
    except Exception as e:
        print(f"测试过程中出错: {str(e)}")

def test_long_text():
    """测试长文本转换"""
    print("\n开始长文本测试...")
    
    # 初始化ChatTTS
    chat = ChatTTS.Chat()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载模型
        chat.load(source='huggingface', device=device)
        
        # 采样说话人
        rand_spk = chat.sample_random_speaker()
        
        # 设置参数 - 使用较慢的语速
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rand_spk,
            temperature=0.6,
            top_P=0.8,
            top_K=25,
        )
        
        params_refine_text = ChatTTS.Chat.RefineTextParams(
            prompt='[oral_1][laugh_0][break_4]',
        )
        
        # 长文本测试
        long_text = """ChatTTS是一个强大的语音合成系统。它使用了先进的深度学习技术，
        能够生成自然、流畅的语音。同时支持多种语音风格和情感表达。"""
        
        # 生成音频
        output_dir = "temp/tts_long"
        os.makedirs(output_dir, exist_ok=True)
        
        wavs = chat.infer(
            [long_text],
            params_refine_text=params_refine_text,
            params_infer_code=params_infer_code,
        )
        
        # 保存音频文件
        output_file = f"{output_dir}/long_text.wav"
        try:
            torchaudio.save(output_file, torch.from_numpy(wavs[0]).unsqueeze(0), 24000)
        except:
            torchaudio.save(output_file, torch.from_numpy(wavs[0]), 24000)
            
        print(f"生成音频: {output_file}")
        if os.path.exists(output_file):
            size = os.path.getsize(output_file)
            print(f"文件大小: {size/1024:.1f}KB")
            
    except Exception as e:
        print(f"长文本测试出错: {str(e)}")

if __name__ == "__main__":
    # 创建必要的目录
    os.makedirs("temp/tts_output", exist_ok=True)
    os.makedirs("temp/tts_long", exist_ok=True)
    
    print("=== ChatTTS 测试开始 ===")
    
    try:
        # 基础测试
        print("\n1. 运行基础语音合成测试")
        test_chattts()
        
        # 长文本测试
        print("\n2. 运行长文本合成测试")
        test_long_text()
        
    except Exception as e:
        print(f"\n测试过程中出错: {str(e)}")
    
    print("\n=== ChatTTS 测试完成 ===")