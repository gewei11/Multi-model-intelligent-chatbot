import os
import logging
from logging.handlers import RotatingFileHandler
import time
from typing import Optional

# 日志格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 默认日志目录
DEFAULT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def setup_logger(name: str = "chatbot", 
                level: str = "INFO", 
                log_file: Optional[str] = None,
                max_size: int = 10 * 1024 * 1024,  # 10MB
                backup_count: int = 5) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别，可选值：DEBUG, INFO, WARNING, ERROR, CRITICAL
        log_file: 日志文件路径，如果为None则使用默认路径
        max_size: 单个日志文件最大大小（字节）
        backup_count: 保留的日志文件数量
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    
    # 设置日志级别
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # 如果已经有处理器，不再添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，创建文件处理器
    if log_file is None:
        # 使用默认日志文件
        os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)
        log_file = os.path.join(DEFAULT_LOG_DIR, f"{name}_{time.strftime('%Y%m%d')}.log")
    else:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 创建文件处理器（支持日志轮转）
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_size, 
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "chatbot") -> logging.Logger:
    """
    获取已配置的日志记录器，如果不存在则创建一个新的
    
    Args:
        name: 日志记录器名称
        
    Returns:
        日志记录器
    """
    logger = logging.getLogger(name)
    
    # 如果记录器没有处理器，则进行配置
    if not logger.handlers:
        return setup_logger(name)
    
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """
    日志适配器，用于添加额外的上下文信息
    """
    def process(self, msg, kwargs):
        # 添加额外的上下文信息
        return f"[{self.extra.get('context', 'UNKNOWN')}] {msg}", kwargs


def get_context_logger(name: str = "chatbot", context: str = "main") -> logging.LoggerAdapter:
    """
    获取带有上下文信息的日志记录器
    
    Args:
        name: 日志记录器名称
        context: 上下文标识，如'core_agent', 'weather_agent'等
        
    Returns:
        带有上下文的日志适配器
    """
    logger = get_logger(name)
    return LoggerAdapter(logger, {"context": context})


# 性能监控日志
class PerformanceMonitor:
    """
    性能监控工具，用于记录函数执行时间等性能指标
    """
    def __init__(self, logger=None, name="performance"):
        self.logger = logger or get_logger(name)
    
    def time_it(self, func_name=None):
        """
        装饰器，用于记录函数执行时间
        
        Args:
            func_name: 函数名称，如果为None则使用被装饰函数的名称
            
        Returns:
            装饰器函数
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                name = func_name or func.__name__
                self.logger.info(f"函数 {name} 执行时间: {execution_time:.4f}秒")
                
                return result
            return wrapper
        return decorator