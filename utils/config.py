import os
import json
import yaml
from typing import Dict, Any

# 默认配置
DEFAULT_CONFIG = {
    "models": {
        "qwen": {
            "model_name": "qwen2.5:7b",
            "api_base": "http://localhost:11434/api/chat",
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "deepseek": {
            "model_name": "MFDoom/deepseek-r1-tool-calling:8b",
            "api_base": "http://localhost:11434/api/chat", 
            "temperature": 0.7,
            "max_tokens": 2048
        },
        "minicpm": {  # 添加 minicpm 配置
            "model_name": "aiden_lu/minicpm-v2.6:Q4_K_M",  # 更正模型名称
            "api_base": "http://localhost:11434/api/chat",
            "temperature": 0.7,
            "max_tokens": 2048,
            "vision": True  # 标记支持视觉功能
        },
        "translation": {  # 添加翻译模型配置
            "system_prompt": """你是一个地名翻译助手。请将给定的中文地名翻译为标准的英文名称。
                只需要回复英文地名，不需要其他解释。
                示例:
                输入: 北京
                输出: Beijing
                输入: 上海
                输出: Shanghai
                现在请翻译: {city}
            """
        }
    },
    "voice": {
        "model_path": "vosk-model-cn-0.22",
        "sample_rate": 16000,
        "enabled": True
    },
    "apis": {
        "weather": {
            "api_key": "<YOUR_WEATHER_API_KEY>",
            "base_url": "https://api.seniverse.com/v3/weather/now.json",
            "forecast_url": "https://api.seniverse.com/v3/weather/daily.json"  # 添加预报接口
        },
        "baidu": {  # 添加百度API配置
            "api_key": "<YOUR_BAIDU_API_KEY>",
            "secret_key": "<YOUR_BAIDU_SECRET_KEY>"
        }
    },
    "logging": {
        "level": "INFO",
        "file": "logs/chatbot.log"
    },
    "domains": {
        "education": {
            "enabled": True
        },
        "ecommerce": {
            "enabled": True
        }
    },
    "conversation": {
        "debug_mode": False,  # 是否显示情感分析调试信息
        "sentiment_enabled": True,  # 是否启用情感分析
        "response_templates": {
            "positive": [
                "很高兴看到您心情不错！{response}",
                "您的积极态度真让人愉快！{response}",
                "太好了！{response}"
            ],
            "negative": [
                "我理解您的心情。{response}",
                "别担心，让我来帮您。{response}",
                "我明白您的感受。{response}"
            ],
            "neutral": [
                "{response}",
                "好的，我明白了。{response}",
                "我来帮您回答。{response}"
            ]
        }
    },
    "features": {
        "sentiment_analysis": {
            "enabled": True,  # 情感分析功能开关
            "show_analysis": True  # 是否显示分析结果
        },
        "weather_enabled": True,
        "education_enabled": True,
        "ecommerce_enabled": True
    }
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    加载配置文件，如果文件不存在则创建默认配置
    
    Args:
        config_path: 配置文件路径，默认为项目根目录下的config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, "config.yaml")
    
    # 如果配置文件不存在，创建默认配置
    if not os.path.exists(config_path):
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(DEFAULT_CONFIG, f, default_flow_style=False, allow_unicode=True)
        return DEFAULT_CONFIG
    
    # 读取配置文件
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.endswith(".json"):
                config = json.load(f)
            else:  # 默认为yaml
                config = yaml.safe_load(f)
        
        # 合并默认配置，确保所有必要的配置项都存在
        merged_config = DEFAULT_CONFIG.copy()
        _deep_update(merged_config, config)
        return merged_config
    
    except Exception as e:
        print(f"加载配置文件失败: {e}，使用默认配置")
        return DEFAULT_CONFIG


def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归更新字典
    
    Args:
        d: 要更新的字典
        u: 更新的内容
        
    Returns:
        更新后的字典
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            _deep_update(d[k], v)
        else:
            d[k] = v
    return d


def save_config(config: Dict[str, Any], config_path: str = None) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 配置文件路径，默认为项目根目录下的config.yaml
    """
    if config_path is None:
        # 获取项目根目录
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(root_dir, "config.yaml")
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            if config_path.endswith(".json"):
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:  # 默认为yaml
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        print(f"保存配置文件失败: {e}")


def get_api_key(service: str, config: Dict[str, Any] = None) -> str:
    """
    获取指定服务的API密钥
    
    Args:
        service: 服务名称，如'qwen', 'deepseek', 'weather'等
        config: 配置字典，如果为None则加载默认配置
        
    Returns:
        API密钥字符串
    """
    if config is None:
        config = load_config()
    
    # 查找API密钥
    if service in ['qwen', 'deepseek']:
        return config.get('models', {}).get(service, {}).get('api_key', '')
    else:
        return config.get('apis', {}).get(service, {}).get('api_key', '')
    
    # 如果环境变量中有设置，优先使用环境变量
    env_key = f"{service.upper()}_API_KEY"
    return os.environ.get(env_key, api_key)