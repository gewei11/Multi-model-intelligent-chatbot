import json
import random
from typing import Dict, Any, List, Optional, Union
import re

from utils.logger import get_logger
from utils.helper_functions import retry, extract_keywords

class EnhancedSentimentAgent:
    """
    增强版情感分析Agent，负责分析用户输入的情感倾向，提供更丰富的情感交互体验
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化增强版情感分析Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("enhanced_sentiment_agent")
        self.logger.info("增强版情感分析Agent初始化")
        
        # 情感分类
        self.emotion_categories = {
            "joy": ["开心", "高兴", "快乐", "兴奋", "愉悦", "喜悦", "欣喜", "满足", "幸福"],
            "sadness": ["难过", "伤心", "悲伤", "痛苦", "忧郁", "沮丧", "失落", "消沉", "哀伤"],
            "anger": ["生气", "愤怒", "恼火", "暴躁", "烦躁", "不满", "恼怒", "气愤", "怒火"],
            "fear": ["害怕", "恐惧", "担忧", "焦虑", "紧张", "惊恐", "忧虑", "不安", "惶恐"],
            "surprise": ["惊讶", "震惊", "意外", "吃惊", "诧异", "惊奇", "惊异", "惊诧", "惊愕"],
            "disgust": ["厌恶", "反感", "恶心", "讨厌", "嫌弃", "憎恶", "鄙视", "蔑视", "不屑"],
            "neutral": ["平静", "中性", "一般", "普通", "正常", "平淡", "无感", "无所谓", "不置可否"]
        }
        
        # 情感强度词汇
        self.intensity_words = {
            "high": ["非常", "极其", "特别", "十分", "格外", "异常", "尤其", "极度", "极为", "无比"],
            "medium": ["很", "相当", "挺", "比较", "蛮", "颇为", "相对", "较为", "不少"],
            "low": ["有点", "稍微", "略微", "一点", "些许", "轻微", "微微", "略为", "少许"]
        }
        
        # 情感回复模板
        self.response_templates = {
            "joy": {
                "high": [
                    "看到您如此开心，我也感到非常高兴！有什么我能帮您做的吗？",
                    "您的喜悦感染了我！能分享是什么让您这么开心吗？",
                    "太棒了！您的好心情真是令人愉快，希望这份快乐能持续下去！"
                ],
                "medium": [
                    "很高兴看到您心情不错！有什么我可以帮您的吗？",
                    "您的积极态度真让人愉快！我会尽力提供最好的服务。",
                    "看到您这么开心，我也感到很愉快！有什么可以分享的吗？"
                ],
                "low": [
                    "看起来您心情不错，有什么我能帮您的吗？",
                    "您似乎有点开心，希望我能让您更加愉快！",
                    "能看到您心情愉快真好，有什么我能为您做的？"
                ]
            },
            "sadness": {
                "high": [
                    "我能感受到您现在非常难过，请记住，这种感觉是暂时的，会好起来的。有什么我能做的来帮助您吗？",
                    "看到您如此伤心，我真的很担心。愿意和我分享发生了什么吗？也许我能提供一些帮助或安慰。",
                    "我很遗憾您正经历这么大的痛苦。请记住，即使在最黑暗的时刻，也总有希望。我在这里陪伴您。"
                ],
                "medium": [
                    "看起来您似乎有些不开心，有什么我能帮到您的吗？",
                    "我注意到您可能遇到了一些困难，请告诉我发生了什么，也许我能提供一些帮助。",
                    "每个人都有不顺心的时候，如果您愿意分享，我很乐意倾听并尝试帮助您。"
                ],
                "low": [
                    "您看起来有点低落，有什么我能做的来帮助您振作起来吗？",
                    "似乎有些事情让您不太开心，需要聊聊吗？",
                    "感觉您的心情有点低落，希望我能帮您找到一些乐趣。"
                ]
            },
            "anger": {
                "high": [
                    "我能理解您现在非常生气，深呼吸可能会有所帮助。如果您愿意，可以告诉我是什么让您如此愤怒？",
                    "看得出您现在非常恼火，我理解这种感受。也许我们可以一起找出解决问题的方法？",
                    "您的愤怒是可以理解的，有时候表达出来是很重要的。如果您想冷静下来，我可以提供一些建议。"
                ],
                "medium": [
                    "我注意到您似乎有些生气，有什么我能帮您解决的问题吗？",
                    "您看起来有点不满，愿意分享是什么让您感到烦恼吗？",
                    "感觉您有些不悦，如果有什么我能帮忙的，请告诉我。"
                ],
                "low": [
                    "您似乎有一点点不满，有什么我可以帮您改善的吗？",
                    "感觉您略微有些不耐烦，有什么我能做的来帮助您吗？",
                    "您看起来有点烦躁，需要我做些什么来帮助您吗？"
                ]
            },
            "fear": {
                "high": [
                    "我能感受到您现在非常担忧，请记住，面对恐惧时，分享和寻求帮助是很重要的。有什么特别让您感到害怕的事情吗？",
                    "看得出您现在非常焦虑，这是很正常的感受。也许我们可以一起分析一下情况，找到一些缓解的方法？",
                    "理解您现在感到非常不安，有时候说出来会让恐惧感减轻一些。愿意和我分享您的担忧吗？"
                ],
                "medium": [
                    "您似乎有些担忧，有什么特别让您感到不安的事情吗？",
                    "感觉您有点焦虑，愿意分享是什么让您感到紧张吗？",
                    "注意到您有些不安，如果您想聊聊，我很乐意倾听。"
                ],
                "low": [
                    "您看起来有一点点担心，有什么我能帮您解决的问题吗？",
                    "感觉您略微有些紧张，需要我提供一些信息或建议吗？",
                    "您似乎有点不安，有什么特别的原因吗？"
                ]
            },
            "surprise": {
                "high": [
                    "哇！看起来您非常震惊！发生了什么让您如此惊讶的事情？",
                    "您看起来非常吃惊！愿意分享是什么让您如此意外吗？",
                    "我能感受到您的惊讶！是什么让您如此震惊呢？"
                ],
                "medium": [
                    "您看起来很惊讶，发生了什么意外的事情吗？",
                    "感觉您有些吃惊，愿意分享是什么让您感到意外吗？",
                    "注意到您有点惊讶，有什么特别的发现吗？"
                ],
                "low": [
                    "您似乎有点意外，有什么让您感到惊讶的事情吗？",
                    "感觉您略微有些诧异，是什么引起了您的注意？",
                    "您看起来有点惊奇，有什么新发现吗？"
                ]
            },
            "disgust": {
                "high": [
                    "我能理解有些事情确实令人非常反感，愿意分享是什么让您如此厌恶吗？",
                    "看得出您对某事非常不满，如果您想讨论，我很乐意倾听。",
                    "理解您现在感到非常不快，有时候表达出来会有所帮助。有什么特别让您感到不适的事情吗？"
                ],
                "medium": [
                    "您看起来对某事有些不满，愿意分享是什么让您感到不快吗？",
                    "感觉您有点反感，有什么特别让您不舒服的事情吗？",
                    "注意到您似乎对某事有些厌恶，如果您想聊聊，我很乐意倾听。"
                ],
                "low": [
                    "您似乎对某事有点不满，有什么让您感到不适的事情吗？",
                    "感觉您略微有些反感，是什么引起了您的不快？",
                    "您看起来有点不悦，有什么我能帮您改善的吗？"
                ]
            },
            "neutral": [
                "有什么我可以帮您的吗？",
                "我在这里随时为您提供帮助。",
                "请告诉我您需要什么服务，我会尽力满足您的需求。",
                "今天有什么特别的问题需要解决吗？",
                "我很乐意为您提供任何帮助，请告诉我您的需求。"
            ]
        }
        
        # 用户情感历史记录
        self.user_emotion_history = []
        
        # 情感变化阈值
        self.emotion_change_threshold = 0.3
    
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
        
        # 分析情感类别和强度
        emotion_result = self._analyze_emotion(user_input)
        self.logger.info(f"情感分析结果: {emotion_result}")
        
        # 记录情感历史
        self.user_emotion_history.append(emotion_result)
        if len(self.user_emotion_history) > 10:  # 只保留最近10条记录
            self.user_emotion_history.pop(0)
        
        # 检测情感变化
        emotion_change = self._detect_emotion_change()
        
        # 根据情感选择回复模板
        response = self._generate_response(emotion_result, emotion_change, user_input, context)
        
        return {
            "emotion": emotion_result["category"],
            "intensity": emotion_result["intensity"],
            "score": emotion_result["score"],
            "emotion_change": emotion_change,
            "response": response
        }
    
    def _analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        分析文本的情感类别和强度
        
        Args:
            text: 输入文本
            
        Returns:
            情感分析结果字典
        """
        # 初始化各情感类别的得分
        emotion_scores = {category: 0 for category in self.emotion_categories.keys()}
        
        # 计算各情感类别的匹配度
        for category, keywords in self.emotion_categories.items():
            for keyword in keywords:
                if keyword in text:
                    emotion_scores[category] += 1
        
        # 确定主要情感类别
        if max(emotion_scores.values()) == 0:
            # 如果没有明确的情感关键词，默认为中性
            category = "neutral"
            score = 0.5
        else:
            # 找出得分最高的情感类别
            category = max(emotion_scores, key=emotion_scores.get)
            score = min(emotion_scores[category] / 3, 1.0)  # 归一化得分，最高为1.0
        
        # 分析情感强度
        intensity = "medium"  # 默认为中等强度
        
        # 检查强度词汇
        for level, words in self.intensity_words.items():
            for word in words:
                if word in text:
                    intensity = level
                    break
            if intensity != "medium":  # 如果已经找到强度词，跳出循环
                break
        
        # 如果是中性情感，不考虑强度
        if category == "neutral":
            intensity = "medium"
        
        return {
            "category": category,
            "score": score,
            "intensity": intensity
        }
    
    def _detect_emotion_change(self) -> Dict[str, Any]:
        """
        检测用户情感变化
        
        Returns:
            情感变化信息
        """
        if len(self.user_emotion_history) < 2:
            return {"detected": False}
        
        # 获取最近两次情感记录
        current = self.user_emotion_history[-1]
        previous = self.user_emotion_history[-2]
        
        # 检查情感类别是否变化
        category_changed = current["category"] != previous["category"]
        
        # 检查情感强度是否显著变化
        score_diff = abs(current["score"] - previous["score"])
        significant_change = score_diff > self.emotion_change_threshold
        
        if category_changed or significant_change:
            return {
                "detected": True,
                "from": previous["category"],
                "to": current["category"],
                "score_change": score_diff,
                "direction": "positive" if current["score"] > previous["score"] else "negative"
            }
        else:
            return {"detected": False}
    
    def _generate_response(self, emotion_result: Dict[str, Any], emotion_change: Dict[str, Any], user_input: str, context: Dict[str, Any]) -> str:
        """
        根据情感分析结果生成回复
        
        Args:
            emotion_result: 情感分析结果
            emotion_change: 情感变化信息
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            生成的回复文本
        """
        category = emotion_result["category"]
        intensity = emotion_result["intensity"]
        
        # 如果检测到明显的情感变化，优先响应情感变化
        if emotion_change.get("detected", False):
            if emotion_change["direction"] == "positive":
                return f"我注意到您的情绪似乎变得更积极了，这真是太好了！{self._get_template_response(category, intensity)}"
            else:
                return f"我感觉您的情绪似乎有些变化，希望一切都好。{self._get_template_response(category, intensity)}"
        
        # 否则，根据当前情感状态生成回复
        return self._get_template_response(category, intensity)
    
    def _get_template_response(self, category: str, intensity: str) -> str:
        """
        从模板中获取回复
        
        Args:
            category: 情感类别
            intensity: 情感强度
            
        Returns:
            模板回复文本
        """
        if category == "neutral":
            # 中性情感不考虑强度
            templates = self.response_templates["neutral"]
        else:
            templates = self.response_templates[category][intensity]
        
        return random.choice(templates)
    
    def get_emotion_categories(self) -> List[str]:
        """
        获取支持的情感类别
        
        Returns:
            情感类别列表
        """
        return list(self.emotion_categories.keys())
    
    def get_user_emotion_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        获取用户情感历史记录
        
        Args:
            limit: 返回的记录数量限制
            
        Returns:
            情感历史记录列表
        """
        return self.user_emotion_history[-limit:] if self.user_emotion_history else []