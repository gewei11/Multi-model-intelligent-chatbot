import json
from typing import Dict, Any, List, Optional, Union

from utils.logger import get_logger
from utils.helper_functions import retry

class SentimentAgent:
    """
    情感分析Agent，负责分析用户输入的情感倾向，并根据情感状态提供相应的回复
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化情感分析Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("sentiment_agent")
        self.logger.info("情感分析Agent初始化")
        
        # 情感回复模板
        self.response_templates = {
            "positive": [
                "很高兴看到您心情不错！有什么我可以帮您的吗？",
                "您的积极态度真让人愉快！我会尽力提供最好的服务。",
                "太好了！您的好心情感染了我，让我们继续愉快的交流吧！"
            ],
            "negative": [
                "看起来您似乎有些不开心，有什么我能帮到您的吗？",
                "我注意到您可能遇到了一些困难，请告诉我发生了什么，也许我能提供一些帮助。",
                "每个人都有不顺心的时候，如果您愿意分享，我很乐意倾听并尝试帮助您。"
            ],
            "neutral": [
                "有什么我可以帮您的吗？",
                "我在这里随时为您提供帮助。",
                "请告诉我您需要什么服务，我会尽力满足您的需求。"
            ]
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        分析用户输入的情感倾向，并生成相应的回复
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            包含情感分析结果和回复的字典
        """
        if context is None:
            context = {}
        
        self.logger.info(f"分析情感: {user_input}")
        
        # 分析情感
        sentiment = self._analyze_sentiment(user_input)
        self.logger.info(f"情感分析结果: {sentiment}")
        
        # 根据情感选择回复模板
        response = self._generate_response(sentiment, user_input, context)
        
        return {
            "sentiment": sentiment,
            "score": self._get_sentiment_score(sentiment),
            "response": response
        }
    
    def _analyze_sentiment(self, text: str) -> str:
        """
        分析文本的情感倾向
        
        Args:
            text: 输入文本
            
        Returns:
            情感类型：positive, negative, neutral
        """
        # 这里使用简单的关键词匹配方法
        # 实际项目中应该使用更复杂的情感分析模型或API
        
        # 积极情感词汇
        positive_words = [
            "开心", "高兴", "快乐", "满意", "喜欢", "爱", "感谢", "谢谢", "好", "棒", 
            "优秀", "赞", "厉害", "不错", "可以", "行", "好的", "嗯", "是的", "对"
        ]
        
        # 消极情感词汇
        negative_words = [
            "不开心", "难过", "伤心", "痛苦", "失望", "生气", "愤怒", "讨厌", "烦", 
            "不好", "差", "糟糕", "坏", "不行", "不可以", "不能", "不要", "滚", "笨", "蠢"
        ]
        
        # 计算情感得分
        positive_score = sum(1 for word in positive_words if word in text)
        negative_score = sum(1 for word in negative_words if word in text)
        
        # 判断情感类型
        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"
    
    def _get_sentiment_score(self, sentiment: str) -> float:
        """
        获取情感得分
        
        Args:
            sentiment: 情感类型
            
        Returns:
            情感得分，范围[-1.0, 1.0]
        """
        if sentiment == "positive":
            return 0.7  # 积极情感得分
        elif sentiment == "negative":
            return -0.7  # 消极情感得分
        else:
            return 0.0  # 中性情感得分
    
    def _generate_response(self, sentiment: str, user_input: str, context: Dict[str, Any]) -> str:
        """
        根据情感类型生成回复
        
        Args:
            sentiment: 情感类型
            user_input: 用户输入
            context: 上下文信息
            
        Returns:
            生成的回复
        """
        import random
        
        # 从对应情感的模板中随机选择一个
        templates = self.response_templates.get(sentiment, self.response_templates["neutral"])
        response = random.choice(templates)
        
        return response
    
    @retry(max_attempts=2)
    def analyze_with_model(self, text: str) -> Dict[str, Any]:
        """
        使用模型进行更复杂的情感分析（示例方法）
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果
        """
        # 这里应该调用更复杂的情感分析模型或API
        # 目前返回模拟数据
        sentiment = self._analyze_sentiment(text)
        
        return {
            "sentiment": sentiment,
            "score": self._get_sentiment_score(sentiment),
            "confidence": 0.85
        }