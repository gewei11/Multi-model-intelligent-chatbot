from typing import Dict, Any
from models.qwen_model import Qwen2Model
from models.deepseek_model import DeepSeekModel

def get_model(config: Dict[str, Any]):
    """
    获取对应的模型实例
    
    Args:
        config: 模型配置
    
    Returns:
        模型实例
    """
    model_type = config.get("type", "qwen").lower()
    
    if model_type == "qwen":
        return Qwen2Model(config)
    elif model_type == "deepseek":
        return DeepSeekModel(config)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
