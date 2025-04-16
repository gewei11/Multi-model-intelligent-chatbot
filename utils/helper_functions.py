import json
import re
import time
import traceback
from typing import Dict, Any, List, Optional, Union, Callable
from functools import wraps

# 异常重试装饰器
def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, exceptions: tuple = (Exception,)):
    """
    异常重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟时间（秒）
        backoff: 延迟时间的增长因子
        exceptions: 需要捕获的异常类型
        
    Returns:
        装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    
                    # 记录异常信息
                    print(f"函数 {func.__name__} 执行失败 (尝试 {attempt}/{max_attempts}): {str(e)}")
                    print(f"等待 {current_delay} 秒后重试...")
                    
                    # 等待后重试
                    time.sleep(current_delay)
                    current_delay *= backoff
        
        return wrapper
    return decorator


# 安全的JSON解析
def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    安全地解析JSON字符串，出错时返回默认值
    
    Args:
        json_str: JSON字符串
        default: 解析失败时返回的默认值
        
    Returns:
        解析后的对象，或默认值
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        print(f"JSON解析失败: {str(e)}")
        return default


# 文本处理函数
def extract_keywords(text: str, min_length: int = 2) -> List[str]:
    """
    从文本中提取关键词（简单实现，实际项目中可能需要更复杂的NLP处理）
    
    Args:
        text: 输入文本
        min_length: 关键词最小长度
        
    Returns:
        关键词列表
    """
    # 移除标点符号和特殊字符
    clean_text = re.sub(r'[^\w\s]', ' ', text)
    
    # 分词并过滤
    words = clean_text.split()
    keywords = [word for word in words if len(word) >= min_length]
    
    # 去重
    return list(set(keywords))


# 安全类型转换
def safe_int(value: Any, default: int = 0) -> int:
    """
    安全地将值转换为整数，出错时返回默认值
    
    Args:
        value: 要转换的值
        default: 转换失败时返回的默认值
        
    Returns:
        转换后的整数，或默认值
    """
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    安全地将值转换为浮点数，出错时返回默认值
    
    Args:
        value: 要转换的值
        default: 转换失败时返回的默认值
        
    Returns:
        转换后的浮点数，或默认值
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


# 异常处理函数
def format_exception(e: Exception) -> str:
    """
    格式化异常信息
    
    Args:
        e: 异常对象
        
    Returns:
        格式化后的异常信息字符串
    """
    return f"{type(e).__name__}: {str(e)}\n{''.join(traceback.format_tb(e.__traceback__))}"


# 缓存装饰器
def simple_cache(ttl: int = 300):
    """
    简单的内存缓存装饰器
    
    Args:
        ttl: 缓存生存时间（秒）
        
    Returns:
        装饰器函数
    """
    cache = {}
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = str(args) + str(sorted(kwargs.items()))
            
            # 检查缓存是否有效
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            return result
        
        # 添加清除缓存的方法
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    return decorator


# 数据验证函数
def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """
    验证字典中是否包含所有必需的字段
    
    Args:
        data: 要验证的字典
        required_fields: 必需的字段列表
        
    Returns:
        缺失的字段列表，如果没有缺失则为空列表
    """
    missing = []
    for field in required_fields:
        if field not in data or data[field] is None:
            missing.append(field)
    return missing


# 字符串处理函数
def truncate_text(text: str, max_length: int = 100, suffix: str = '...') -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后添加的后缀
        
    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix