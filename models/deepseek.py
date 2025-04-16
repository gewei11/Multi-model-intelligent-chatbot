import json
import requests
from typing import Dict, Any, List, Optional, Union, Iterator

from utils.logger import get_logger
from utils.helper_functions import retry

class DeepSeekModel:
    """
    DeepSeek模型接口，封装对DeepSeek模型的调用
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化DeepSeek模型接口
        """
        self.config = config
        self.logger = get_logger("deepseek_model")
        self.logger.info("DeepSeek模型接口初始化")
        
        # 获取API密钥和模型名称 (本地 Ollama 不需要 API 密钥)
        self.model_name = config.get("model_name", "MFDoom/deepseek-r1-tool-calling:8b")
        
        # 默认参数
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "max_tokens": config.get("max_tokens", 2048),
            "top_p": config.get("top_p", 0.8),
        }
        
        # API端点
        self.api_base = config.get("api_base", "http://localhost:11434/api/chat")
    
    @retry(max_attempts=3, delay=1.0)
    def generate(self, prompt: str, conversation_history: List[Dict[str, str]] = None, params: Dict[str, Any] = None) -> Union[str, Iterator[str]]:
        """
        生成文本，支持流式输出
        """
        self.logger.info(f"使用DeepSeek模型生成文本")
        
        try:
            # 构建消息历史
            messages = []
            if conversation_history:
                messages.extend(conversation_history)
            
            if not messages or messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": prompt})
            
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "stream": True,  # 启用流式输出
                "options": {
                    "temperature": self.default_params["temperature"],
                    "num_predict": self.default_params["max_tokens"]
                }
            }
            
            # 发送请求到 Ollama API
            response = requests.post(
                self.api_base,
                headers={"Content-Type": "application/json"},
                json=request_data,
                stream=True  # 启用流式响应
            )
            response.raise_for_status()
            
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
            self.logger.error(f"DeepSeek模型生成文本时发生错误: {str(e)}")
            yield f"抱歉，使用DeepSeek模型时发生错误: {str(e)}"
    
    @retry(max_attempts=3, delay=1.0)
    def function_call(self, prompt: str, functions: List[Dict[str, Any]], conversation_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        调用函数
        
        Args:
            prompt: 提示文本
            functions: 函数定义列表
            conversation_history: 对话历史
            
        Returns:
            函数调用结果
        """
        self.logger.info(f"使用DeepSeek模型进行函数调用")
        
        # 检查API密钥
        if not self.api_key:
            self.logger.error("未配置DeepSeek API密钥")
            return {"error": "未配置API密钥"}
        
        try:
            # 构建消息列表
            messages = []
            
            # 添加对话历史
            if conversation_history:
                for msg in conversation_history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    
                    # 转换角色名称（如果需要）
                    if role == "user":
                        role = "user"
                    elif role == "assistant":
                        role = "assistant"
                    elif role == "system":
                        role = "system"
                    
                    if role and content:
                        messages.append({"role": role, "content": content})
            
            # 添加当前提示
            if not messages or messages[-1]["role"] != "user":
                messages.append({"role": "user", "content": prompt})
            
            # 构建请求数据
            request_data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.default_params["temperature"],
                "max_tokens": self.default_params["max_tokens"],
                "functions": functions,
                "function_call": "auto"
            }
            
            # 设置请求头
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # 发送请求
            response = requests.post(self.api_base, headers=headers, json=request_data)
            response.raise_for_status()  # 检查请求是否成功
            
            # 解析响应
            result = response.json()
            
            # 提取函数调用结果
            function_call = result.get("choices", [{}])[0].get("message", {}).get("function_call", {})
            
            self.logger.info("DeepSeek模型函数调用成功")
            return function_call
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"DeepSeek API请求失败: {str(e)}")
            return {"error": str(e)}
        except Exception as e:
            self.logger.error(f"DeepSeek模型函数调用时发生错误: {str(e)}")
            return {"error": str(e)}