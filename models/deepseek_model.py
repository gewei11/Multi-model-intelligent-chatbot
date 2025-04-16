from typing import Dict, Any, List, Iterator
import requests
import json

class DeepSeekModel:
    def __init__(self, config: Dict[str, Any]):
        self.api_base = config.get("api_base", "http://localhost:11434/v1/chat/completions")
        self.model_name = config.get("model_name", "deepseek")
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 2048)

    def generate(self, prompt: str, messages: List[Dict[str, str]] = None) -> Iterator[str]:
        """生成回复"""
        if messages is None:
            messages = []
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                self.api_base,
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "stream": True
                },
                stream=True
            )
            
            # 逐行读取响应
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("message", {}).get("content"):
                            yield chunk["message"]["content"]
                    except json.JSONDecodeError:
                        continue
                    
        except Exception as e:
            yield f"生成回复失败: {str(e)}"
