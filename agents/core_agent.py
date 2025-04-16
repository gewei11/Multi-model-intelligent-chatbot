import json
import time
from typing import Dict, Any, List, Optional, Union, Iterator, BinaryIO
import re

from utils.logger import get_logger
from utils.helper_functions import extract_keywords, retry
from tools.weather_tools import WeatherQueryTool

# 导入各个功能性Agent
from agents.conversation_agent import ConversationAgent
from agents.weather_agent import WeatherAgent
from agents.voice_agent import VoiceAgent  # 取消注释
# from agents.sentiment_agent import SentimentAgent
# from agents.domain_agents.education_agent import EducationAgent
from agents.domain_agents.ecommerce_agent import EcommerceAgent  # 取消注释该行

class CoreAgent:
    """
    核心调度Agent，负责解析用户输入并路由到合适的子Agent
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化核心Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("core_agent")
        self.logger.info("核心Agent初始化")
        
        # 初始化对话历史
        self.conversation_history = []
        
        # 添加工具定义
        self.weather_tool = WeatherQueryTool()
        self.tools = [
            {
                "type": "function",
                "function": self.weather_tool.parameters,
                "name": self.weather_tool.name,
                "description": self.weather_tool.description
            }
        ]
        
        # 初始化各个功能性Agent
        self.conversation_agent = ConversationAgent(config)
        
        # 在初始化 WeatherAgent 之前，添加模型到配置
        config["llm_model"] = self.conversation_agent.deepseek_model  # 或使用 qwen_model
        self.weather_agent = WeatherAgent(config)
        
        # 初始化语音Agent
        self.voice_agent = VoiceAgent(config)  # 取消注释并启用
        
        # self.sentiment_agent = SentimentAgent(config)
        from agents.domain_agents.education_agent import EducationAgent
        self.education_agent = EducationAgent(config)  # 初始化教育Agent(数学Agent)
        self.ecommerce_agent = EcommerceAgent(config)  # 电商Agent
        from agents.domain_agents.government_agent import GovernmentAgent
        self.government_agent = GovernmentAgent(config)  # 初始化政务Agent
        
        # 初始化Agent路由表
        self.agent_router = {
            "conversation": self._route_to_conversation,
            "weather": self._route_to_weather,
            "voice": self._route_to_voice,
            "sentiment": self._route_to_sentiment,
            "education": self._route_to_education,
            "ecommerce": self._route_to_ecommerce,
            "government": self._route_to_government
        }
        
        # 添加或修改路由规则
        self.routing_rules = {
            "weather": {
                "keywords": ["天气", "气温", "温度", "下雨", "阴天", "晴天", "多云"],
                "patterns": [
                    r".*?(?:查询|查看|想知道|告诉我)?.*?(?:的)?天气.*?",
                    r".*?天气(?:怎么样|如何).*?"
                ]
            },
            "ecommerce": {
                "keywords": ["购物", "商品", "价格", "订单", "购买", "电商", "手机", "电脑", "耳机", "平板"],
                "patterns": [
                    r".*?(?:想买|推荐|查询|搜索).*?(?:商品|产品|手机|电脑|耳机|平板).*?",
                    r".*?(?:\d+元|\d+到\d+元).*?(?:以下|以内|之间).*?",
                    r".*?(?:购物指南|选购指南|购买建议).*?",
                    r".*?(?:查询|查看).*?(?:订单).*?"
                ]
            },
            "education": {
                "keywords": ["数学", "计算", "方程", "函数", "几何", "代数", "微积分", "物理", "化学", "教育"],
                "patterns": [
                    r".*?(?:计算|求解|证明|解方程).*?",
                    r".*?(?:\d+[+\-*/^]\d+).*?",
                    r".*?(?:数学|物理|化学).*?(?:问题|题目|公式).*?"
                ]
            },
            "government": {
                "keywords": ["政务", "证件", "社保", "医保", "公积金", "税务", "户口", "驾照", "身份证", "护照"],
                "patterns": [
                    r".*?(?:办理|申请|查询).*?(?:证件|社保|医保|公积金|税务).*?",
                    r".*?(?:政府|政务).*?(?:服务|咨询).*?"
                ]
            },
            "conversation": {
                "keywords": ["你好", "请问", "帮我", "谢谢"],
                "patterns": []
            },
            # ...other rules...
        }
    
    def process_input(self, user_input: str, context: Dict[str, Any] = None) -> Union[str, Iterator[str]]:
        """
        处理用户输入，支持流式输出
        """
        if context is None:
            context = {}
        
        self.logger.info(f"收到用户输入: {user_input}")
        
        # 记录用户输入到对话历史
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # 分析用户输入，确定应该路由到哪个Agent
        agent_type = self._route_to_agent(user_input, context)
        self.logger.info(f"路由到Agent类型: {agent_type}")
        
        # 路由到对应的Agent处理
        if agent_type in self.agent_router:
            response_iterator = self.agent_router[agent_type](user_input, context)
        else:
            # 默认路由到对话Agent
            response_iterator = self._route_to_conversation(user_input, context)
        
        # 收集完整响应以添加到历史记录
        full_response = ""
        for chunk in response_iterator:
            full_response += chunk
            yield chunk
        
        # 记录回复到对话历史
        self.conversation_history.append({"role": "assistant", "content": full_response})
    
    def _route_to_agent(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """路由到合适的Agent"""
        if context is None:
            context = {}
            
        # 优化路由逻辑
        for agent_type, rules in self.routing_rules.items():
            # 如果是天气Agent且天气功能被禁用，则跳过
            if agent_type == "weather" and context.get("weather_enabled") is False:
                continue
                
            # 检查关键词
            if any(keyword in user_input for keyword in rules["keywords"]):
                # 再次检查天气功能是否启用
                if agent_type == "weather" and context.get("weather_enabled") is False:
                    self.logger.info("天气功能已禁用，跳过路由到weather agent")
                    continue
                self.logger.info(f"根据关键词路由到Agent类型: {agent_type}")
                return agent_type
                
            # 检查正则表达式模式
            for pattern in rules["patterns"]:
                if re.search(pattern, user_input):
                    # 再次检查天气功能是否启用
                    if agent_type == "weather" and context.get("weather_enabled") is False:
                        self.logger.info("天气功能已禁用，跳过路由到weather agent")
                        continue
                    self.logger.info(f"根据模式匹配路由到Agent类型: {agent_type}")
                    return agent_type
        
        # 如果包含天气相关词汇，优先路由到weather agent（但需检查天气功能是否启用）
        if "天气" in user_input and context.get("weather_enabled", True):
            self.logger.info("检测到天气查询，路由到weather agent")
            return "weather"
            
        # 默认路由到对话agent
        self.logger.info("使用默认路由到conversation agent")
        return "conversation"
    
    def _route_to_conversation(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到对话Agent处理
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            处理结果
        """
        return self.conversation_agent.process(user_input, self.conversation_history, context)
    
    def _route_to_weather(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到天气Agent处理
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            处理结果
        """
        try:
            # 使用工具解析查询
            params = self.weather_tool.parse_query(user_input)
            if not params["city"]:
                return "抱歉，我没有识别出您想查询哪个城市的天气。请明确指定城市名称。"
            
            # 调用天气查询
            return self.weather_agent.process(user_input, {
                **context,
                "parsed_params": params  # 传递解析后的参数
            })
            
        except Exception as e:
            self.logger.error(f"处理天气查询时发生错误: {str(e)}")
            return f"抱歉，处理天气查询时发生错误: {str(e)}"
    
    def _route_to_voice(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到语音Agent处理
        
        Args:
            user_input: 用户输入文本（可能是语音文件路径）
            context: 上下文信息
            
        Returns:
            处理结果
        """
        # 这里会在实现VoiceAgent后取消注释
        # return self.voice_agent.process(user_input, context)
        
        # 临时返回
        return f"语音Agent处理中...\n\n这是一个临时回复，实际项目中会调用语音识别模型处理。"
    
    def _route_to_sentiment(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到情感分析Agent处理
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            处理结果
        """
        # 这里会在实现SentimentAgent后取消注释
        # return self.sentiment_agent.process(user_input, context)
        
        # 临时返回
        return f"情感分析Agent处理中...\n\n您的输入是：{user_input}\n\n这是一个临时回复，实际项目中会分析情感并给出相应回复。"
    
    def _route_to_education(self, user_input: str, context: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """
        路由到教育Agent处理
        """
        try:
            # 使用教育Agent处理数学等教育相关查询
            result = self.education_agent.process(user_input, context)
            if not result["success"]:
                return f"抱歉，处理遇到问题: {result.get('error', '未知错误')}"
                
            # 直接返回response内容
            return result["response"]
        except Exception as e:
            self.logger.error(f"处理教育查询时发生错误: {str(e)}")
            return f"抱歉，处理教育查询时发生错误: {str(e)}"
    
    def _route_to_ecommerce(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到电商Agent处理
        """
        try:
            result = self.ecommerce_agent.process(user_input, context)
            if not result["success"]:
                return f"抱歉，处理遇到问题: {result.get('error', '未知错误')}"
                
            # 直接返回response内容
            return result["response"]
            
        except Exception as e:
            self.logger.error(f"处理电商查询时发生错误: {str(e)}")
            return f"抱歉，处理失败: {str(e)}"
    
    def process_voice_input(self, audio_data: Union[str, bytes, BinaryIO], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """处理语音输入的完整流程"""
        if context is None:
            context = {}

        try:
            # 1. 语音识别和初步情感分析
            voice_result = self.voice_agent.process_input(audio_data, context)
            if "error" in voice_result:
                return voice_result

            recognized_text = voice_result["recognized_text"]
            sentiment_result = voice_result.get("sentiment_result")

            # 2. 生成回复
            base_response = ""
            for chunk in self.conversation_agent.process(
                recognized_text, 
                self.conversation_history,
                {**context, "sentiment_result": sentiment_result}
            ):
                base_response += chunk

            # 3. 根据情感调整回复（如果开启）
            if context.get("sentiment_enabled", True) and sentiment_result:
                from tools.sentiment_tools import SentimentAnalysisTool
                sentiment_tool = SentimentAnalysisTool(self.config)
                adjusted_response = sentiment_tool.adjust_response(base_response, sentiment_result)
            else:
                adjusted_response = base_response

            # 4. 生成语音回复
            try:
                speech_file = self.voice_agent.synthesize_speech(adjusted_response)
            except Exception as e:
                self.logger.error(f"语音合成失败: {str(e)}")
                speech_file = None

            return {
                "recognized_text": recognized_text,
                "sentiment_result": sentiment_result,
                "text_response": adjusted_response,
                "speech_file": speech_file
            }

        except Exception as e:
            self.logger.error(f"处理语音输入失败: {str(e)}")
            return {"error": f"处理失败: {str(e)}"}
    
    def _route_to_government(self, user_input: str, context: Dict[str, Any]) -> str:
        """
        路由到政务Agent处理
        """
        try:
            result = self.government_agent.process(user_input, context)
            if not result["success"]:
                return f"抱歉，处理遇到问题: {result.get('error', '未知错误')}"
                
            # 直接返回response内容
            return result["response"]
            
        except Exception as e:
            self.logger.error(f"处理政务查询时发生错误: {str(e)}")
            return f"抱歉，处理失败: {str(e)}"
    
    def clear_history(self) -> None:
        """
        清除对话历史
        """
        self.conversation_history = []
        self.logger.info("对话历史已清除")