import requests
import json
import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from utils.logger import get_logger
from utils.helper_functions import retry

class SentimentAnalysisTool:
    """情感分析工具"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("sentiment_tool")
        self.api_key = config.get("apis", {}).get("baidu", {}).get("api_key")
        self.secret_key = config.get("apis", {}).get("baidu", {}).get("secret_key")
        self.access_token = None

        # 情绪标签映射
        self.emotion_map = {
            "optimistic": "乐观",
            "pessimistic": "悲观",
            "neutral": "中性",
            # 子情绪映射
            "happy": "开心",
            "angry": "愤怒",
            "sad": "伤心",
            "disgusting": "厌恶",
            "fearful": "恐惧"
        }
        
        # 情感响应模板
        self.response_templates = {
            "positive": [
                "很高兴看到您心情不错！{response}",
                "您的积极态度真让人愉快！{response}",
                "太好了！{response}"
            ],
            "negative": [
                "理解您的心情。{response}",
                "别担心，让我来帮您。{response}",
                "我明白您的感受。{response}"
            ],
            "neutral": [
                "{response}",
                "好的，我明白了。{response}",
                "我来帮您回答。{response}"
            ]
        }

    @retry(max_attempts=3, delay=1.0)
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """分析文本情感倾向"""
        try:
            if not self.access_token:
                self.access_token = self._get_access_token()
            
            url = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={self.access_token}"
            
            payload = json.dumps({"text": text}, ensure_ascii=False)
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
            response.raise_for_status()
            result = response.json()
            
            if "items" in result and len(result["items"]) > 0:
                item = result["items"][0]
                sentiment_map = {0: "负面", 1: "中性", 2: "正面"}
                return {
                    "sentiment": sentiment_map.get(item.get("sentiment"), "未知"),
                    "confidence": item.get("confidence", 0),
                    "positive_prob": item.get("positive_prob", 0),
                    "negative_prob": item.get("negative_prob", 0)
                }
            return {"error": "未能获取情感分析结果"}
            
        except Exception as e:
            self.logger.error(f"情感分析失败: {str(e)}")
            return {"error": f"情感分析失败: {str(e)}"}
    
    @retry(max_attempts=3, delay=1.0)
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """分析对话情绪"""
        try:
            if not self.access_token:
                self.access_token = self._get_access_token()
            
            url = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/emotion?charset=UTF-8&access_token={self.access_token}"
            
            payload = json.dumps({"text": text}, ensure_ascii=False)
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            response = requests.post(url, headers=headers, data=payload.encode("utf-8"))
            response.raise_for_status()
            result = response.json()
            
            if "items" in result and len(result["items"]) > 0:
                # 情绪映射
                emotion_map = {
                    "optimistic": "乐观",
                    "pessimistic": "悲观",
                    "neutral": "中性",
                    "angry": "愤怒",
                    "happy": "开心",
                    "sad": "伤心",
                    "disgusting": "厌恶",
                    "fearful": "恐惧"
                }
                
                # 找出主要情绪（概率最高的）
                main_emotion = max(result["items"], key=lambda x: x.get("prob", 0))
                
                # 处理情绪分析结果
                emotions = []
                for item in result["items"]:
                    emotion = {
                        "type": emotion_map.get(item.get("label"), item.get("label", "未知")),
                        "probability": item.get("prob", 0),
                        "sub_emotions": [],
                        "replies": item.get("replies", [])
                    }
                    
                    # 处理子情绪
                    if "subitems" in item and item["subitems"]:
                        for subitem in item["subitems"]:
                            sub_emotion = {
                                "type": emotion_map.get(subitem.get("label"), subitem.get("label", "未知")),
                                "probability": subitem.get("prob", 0)
                            }
                            emotion["sub_emotions"].append(sub_emotion)
                    
                    emotions.append(emotion)
                
                # 构建返回结果
                return {
                    "main_emotion": emotion_map.get(main_emotion.get("label"), main_emotion.get("label", "未知")),
                    "confidence": main_emotion.get("prob", 0),
                    "detailed_emotions": emotions,
                    "text": result.get("text", text),
                    "suggested_replies": main_emotion.get("replies", [])
                }
            
            return {"error": "未能获取情绪分析结果"}
            
        except Exception as e:
            self.logger.error(f"情绪分析失败: {str(e)}")
            return {"error": f"情绪分析失败: {str(e)}"}
    
    @retry(max_attempts=3, delay=1.0)
    def analyze_combined(self, text: str) -> Dict[str, Any]:
        """综合情感分析"""
        # 简单实现，实际项目中应该使用更复杂的情感分析模型
        sentiment = self._analyze_basic_sentiment(text)
        emotion = self._analyze_emotion(text)
        
        return {
            "sentiment_analysis": sentiment,
            "emotion_analysis": emotion,
            "conclusion": self._generate_conclusion(sentiment, emotion)
        }

    def adjust_response(self, base_response: str, sentiment_result: Dict[str, Any]) -> str:
        """根据情感分析结果调整回复"""
        sentiment = sentiment_result["sentiment_analysis"]["sentiment"].lower()
        template = random.choice(self.response_templates.get(sentiment, self.response_templates["neutral"]))
        return template.format(response=base_response)

    def _analyze_basic_sentiment(self, text: str) -> Dict[str, Any]:
        """基础情感倾向分析"""
        # 简单的关键词匹配实现
        positive_words = ["开心", "高兴", "快乐", "好", "棒", "喜欢"]
        negative_words = ["难过", "伤心", "不好", "讨厌", "失望"]
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return {
                "sentiment": "neutral",
                "confidence": 0.6,
                "positive_prob": 0.33,
                "negative_prob": 0.33
            }
            
        positive_prob = positive_count / (total)
        
        if positive_prob > 0.6:
            sentiment = "positive"
            confidence = min(positive_prob + 0.2, 0.99)
        elif positive_prob < 0.4:
            sentiment = "negative"
            confidence = min((1 - positive_prob) + 0.2, 0.99)
        else:
            sentiment = "neutral"
            confidence = 0.6
            
        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_prob": positive_prob,
            "negative_prob": 1 - positive_prob
        }

    def _analyze_emotion(self, text: str) -> Dict[str, Any]:
        """详细情绪分析"""
        emotions = {
            "乐观": 0.0,
            "开心": 0.0,
            "中性": 0.0,
            "悲观": 0.0
        }
        
        # 简单实现，实际项目中应使用更复杂的情绪分析模型
        if any(word in text for word in ["开心", "高兴", "快乐"]):
            emotions["开心"] = 0.8
            emotions["乐观"] = 0.7
        elif any(word in text for word in ["难过", "伤心", "不好"]):
            emotions["悲观"] = 0.8
        else:
            emotions["中性"] = 0.8
            
        # 找出主要情绪
        main_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return {
            "main_emotion": main_emotion[0],
            "confidence": main_emotion[1],
            "detailed_emotions": [
                {"type": k, "probability": v}
                for k, v in emotions.items()
                if v > 0
            ]
        }

    def _generate_conclusion(self, sentiment: Dict[str, Any], emotion: Dict[str, Any]) -> str:
        """生成综合结论"""
        return f"这句话整体表现为{sentiment['sentiment']}情感({sentiment['confidence']:.1%}置信度), 主要情绪是{emotion['main_emotion']}({emotion['confidence']:.1%}置信度)。"

    def _get_access_token(self) -> str:
        """获取百度API访问令牌"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        try:
            response = requests.post(url, params=params)
            response.raise_for_status()
            return str(response.json().get("access_token"))
        except Exception as e:
            self.logger.error(f"获取access_token失败: {str(e)}")
            raise
