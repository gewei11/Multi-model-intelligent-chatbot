import sys
import os
from PIL import Image
import io
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.minicpm import MiniCPMModel
from utils.config import load_config

def test_image_analysis():
    """测试图片分析功能"""
    # 配置
    config = {
        "models": {
            "minicpm": {
                "model_name": "aiden_lu/minicpm-v2.6:Q4_K_M",
                "api_base": "http://localhost:11434/v1/chat/completions",
                "temperature": 0.7,
                "max_tokens": 2048,
                "vision": True
            }
        }
    }
    
    # 创建模型实例
    model = MiniCPMModel(config["models"]["minicpm"])
    
    # 测试图片分析
    image_path = r"C:\Users\23668\Pictures\首页效果图.png"
    prompt = "这张图片里有什么"
    
    print(f"正在分析图片：{image_path}")
    print(f"提示：{prompt}")
    print("-" * 50)
    
    try:
        # 使用流式输出打印结果
        print("分析结果：")
        full_response = ""
        for chunk in model.generate(prompt, images=[image_path]):
            print(chunk, end="", flush=True)
            full_response += chunk
        print("\n")
            
    except Exception as e:
        print(f"分析失败：{str(e)}")

if __name__ == "__main__":
    test_image_analysis()
