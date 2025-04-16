import json
import base64
import requests
from typing import Dict, Any, List, Optional, Union, Iterator
from pathlib import Path
from PIL import Image, ImageOps  # 添加 PIL 导入
import io

from utils.logger import get_logger

class MiniCPMModel:
    """MiniCPM 模型接口，支持多模态对话"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("minicpm_model")
        self.logger.info("MiniCPM模型接口初始化")
        
        self.model_name = config.get("model_name", "aiden_lu/minicpm-v2.6:Q4_K_M")
        self.api_base = config.get("api_base", "http://localhost:11434/v1/chat/completions")
        
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 2048)
        }

    def _encode_image(self, image_path: str) -> str:
        """将图片转换为 base64 编码"""
        try:
            with Image.open(image_path) as img:
                # 确保图片是 RGB 模式
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # 转换为 bytes
                buffered = io.BytesIO()
                img.save(buffered, format="PNG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')
                
        except Exception as e:
            self.logger.error(f"图片编码失败: {str(e)}")
            raise

    def generate(self, prompt: str, images: List[str] = None) -> Iterator[str]:
        """生成回复，支持图片输入"""
        try:
            # 构建消息内容
            content = []
            if images:
                for img_path in images:
                    if Path(img_path).exists():
                        base64_img = self._encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{base64_img}"
                        })
            content.append({"type": "text", "text": prompt})

            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "stream": False  # 设置为 False 以获取完整响应
            }

            # 发送请求
            response = requests.post(
                self.api_base,
                headers={"Content-Type": "application/json"},
                json=request_data
            )
            response.raise_for_status()

            # 解析响应
            result = response.json()
            if isinstance(result, dict) and "choices" in result:
                content = result["choices"][0]["message"]["content"]
                yield content
            else:
                self.logger.error(f"意外的响应格式: {result}")
                yield "抱歉，处理响应时出现错误。"

        except Exception as e:
            self.logger.error(f"MiniCPM模型生成回复时发生错误: {str(e)}")
            yield f"抱歉，使用MiniCPM模型时发生错误: {str(e)}"

    def analyze_image(self, image_path: str, prompt: str = "这张图片里有什么") -> str:
        """分析单张图片"""
        return "".join(self.generate(prompt, images=[image_path]))
