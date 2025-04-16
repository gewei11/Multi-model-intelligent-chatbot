import json
import time
from typing import Dict, Any, List, Optional, Union, Iterator

from utils.logger import get_logger
from utils.helper_functions import retry, safe_json_loads

# 导入模型接口
from models.qwen2_5 import Qwen2Model
from models.deepseek import DeepSeekModel

class ConversationAgent:
    """
    对话Agent，负责处理用户的对话请求，集成Qwen2和DeepSeek模型
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化对话Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("conversation_agent")
        self.logger.info("对话Agent初始化")
        
        # 初始化模型
        self.qwen_model = Qwen2Model(config["models"]["qwen"])
        self.deepseek_model = DeepSeekModel(config["models"]["deepseek"])
        
        # 添加情感分析工具
        from tools.sentiment_tools import SentimentAnalysisTool
        self.sentiment_tool = SentimentAnalysisTool(config)
        
        # 情感响应模板
        self.sentiment_templates = {
            "positive": [
                "看得出来您心情不错！{response}",
                "很高兴看到您这么开心！{response}",
                "您的好心情感染了我！{response}"
            ],
            "negative": [
                "理解您的心情。{response}",
                "别担心，让我们一起来解决这个问题。{response}",
                "我明白您的感受。{response}"
            ],
            "neutral": [
                "{response}",
                "好的，我明白了。{response}",
                "我来帮您解答。{response}"
            ]
        }
        
        # 模型选择策略
        self.model_selection_strategies = {
            "自动（智能选择）": self._auto_select_model,
            "Qwen2.5": self._use_qwen_model,
            "DeepSeek": self._use_deepseek_model,
            "混合模式": self._use_hybrid_model
        }
    
    def process(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """处理用户输入，生成回复"""
        self.logger.info(f"处理用户输入: {user_input}")
        
        try:
            # 检查情感分析功能是否启用
            sentiment_enabled = context.get("sentiment_enabled", True)
            show_analysis = context.get("show_analysis", True)
            
            if sentiment_enabled:
                # 进行情感分析
                sentiment_result = self.sentiment_tool.analyze_combined(user_input)
                analysis_report = self._format_sentiment_analysis(sentiment_result) if show_analysis else ""
            else:
                sentiment_result = None
                analysis_report = ""
            
            # 使用选定的模型生成基础回复
            strategy = self.model_selection_strategies[context.get("model_option", "自动（智能选择）")]
            base_response = ""
            for chunk in strategy(user_input, conversation_history, context):
                base_response += chunk
            
            # 根据情感分析结果调整回复（如果启用）
            if sentiment_enabled:
                adjusted_response = self._adjust_response_with_sentiment(base_response, sentiment_result)
            else:
                adjusted_response = base_response
            
            # 组合完整响应
            if sentiment_enabled and show_analysis and analysis_report:
                full_response = f"{analysis_report}\n\n模型回答：\n{adjusted_response}"
            else:
                full_response = adjusted_response
            
            # 流式返回结果
            for char in full_response:
                yield char
            
        except Exception as e:
            self.logger.error(f"处理失败: {str(e)}")
            yield f"抱歉，处理您的输入时遇到了问题: {str(e)}"

    def _format_sentiment_analysis(self, result: Dict[str, Any]) -> str:
        """格式化情感分析结果"""
        try:
            # 提取情感分析结果
            sentiment = result["sentiment_analysis"]
            emotion = result["emotion_analysis"]
            
            # 构建输出字符串
            output = [
                "情感分析结果：",
                "\n1. 情感倾向分析：",
                f"情感倾向: {sentiment['sentiment']}",
                f"置信度: {sentiment['confidence']:.2%}",
                f"积极概率: {sentiment['positive_prob']:.2%}",
                f"消极概率: {sentiment['negative_prob']:.2%}",
                
                "\n2. 对话情绪分析：",
                f"主要情绪: {emotion['main_emotion']}",
                f"置信度: {emotion['confidence']:.2%}",
                
                "\n详细情绪分布："
            ]
            
            # 添加详细情绪分布
            for e in emotion["detailed_emotions"]:
                output.append(f"- {e['type']}: {e['probability']:.2%}")
                if e.get("sub_emotions"):
                    for sub_e in e["sub_emotions"]:
                        output.append(f"  - {sub_e['type']}: {sub_e['probability']:.2%}")
            
            # 添加综合结论
            if "conclusion" in result:
                output.extend(["\n综合结论：", result["conclusion"]])
            
            return "\n".join(output)
            
        except Exception as e:
            self.logger.error(f"格式化情感分析结果失败: {str(e)}")
            return "情感分析结果格式化失败"
    
    def _adjust_response_with_sentiment(self, base_response: str, sentiment_result: Dict[str, Any]) -> str:
        """根据情感分析结果调整回复"""
        import random
        
        try:
            # 获取主要情感倾向
            sentiment = sentiment_result["sentiment_analysis"]["sentiment"]
            
            # 获取情感模板
            templates = self.sentiment_templates.get(sentiment.lower(), self.sentiment_templates["neutral"])
            template = random.choice(templates)
            
            # 添加情感分析信息到上下文
            context_info = []
            
            # 添加情感倾向信息
            context_info.append(f"[情感倾向: {sentiment}, "
                             f"置信度: {sentiment_result['sentiment_analysis']['confidence']:.1%}]")
            
            # 添加主要情绪信息
            emotion = sentiment_result["emotion_analysis"]["main_emotion"]
            context_info.append(f"[主要情绪: {emotion}, "
                             f"置信度: {sentiment_result['emotion_analysis']['confidence']:.1%}]")
            
            # 如果有建议回复，随机选择一个
            suggested_replies = sentiment_result["emotion_analysis"].get("suggested_replies", [])
            if suggested_replies:
                context_info.append(f"[建议回复: {random.choice(suggested_replies)}]")
            
            # 组合最终回复
            final_response = template.format(response=base_response)
            
            # 在开发模式下添加情感分析信息（可通过配置控制是否显示）
            if self.config.get("debug_mode", False):
                final_response = "\n".join(context_info) + "\n\n" + final_response
            
            return final_response
            
        except Exception as e:
            self.logger.error(f"调整回复失败: {str(e)}")
            return base_response  # 如果处理失败，返回原始回复
    
    def _auto_select_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        """
        自动选择合适的模型
        
        Args:
            user_input: 用户输入文本
            conversation_history: 对话历史
            context: 上下文信息
            
        Returns:
            生成的回复
        """
        # 分析用户输入，决定使用哪个模型
        # 这里实现一个简单的规则：
        # - 如果包含专业术语或复杂问题，使用DeepSeek
        # - 否则使用Qwen2.5（通常响应更快）
        
        # 专业领域关键词
        professional_keywords = [
            "算法", "编程", "代码", "科学", "研究", "论文", "医学", "法律", 
            "金融", "投资", "经济学", "物理", "化学", "生物", "数学"
        ]
        
        # 检查是否包含专业关键词
        if any(keyword in user_input for keyword in professional_keywords):
            self.logger.info("检测到专业问题，使用DeepSeek模型")
            return self._use_deepseek_model(user_input, conversation_history, context)
        else:
            self.logger.info("使用Qwen2.5模型处理一般问题")
            return self._use_qwen_model(user_input, conversation_history, context)
    
    def _use_qwen_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> str:
        """
        使用Qwen2.5模型生成回复
        
        Args:
            user_input: 用户输入文本
            conversation_history: 对话历史
            context: 上下文信息
            
        Returns:
            生成的回复
        """
        return self.qwen_model.generate(user_input, conversation_history)
    
    def _use_deepseek_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """
        使用DeepSeek模型生成专业知识回复
        """
        # 添加系统提示来增强教育相关回答的质量
        system_prompt = {
            "role": "system", 
            "content": "你是一个专业的教育助手，擅长解答学术和专业知识相关的问题。请提供准确、清晰且结构化的回答。"
        }
        
        # 在对话历史开头添加系统提示
        messages = [system_prompt]
        if conversation_history:
            messages.extend(conversation_history)
        
        return self.deepseek_model.generate(user_input, messages)
    
    def _use_hybrid_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Union[str, Iterator[str]]:
        """使用混合模式，结合两个模型的优势"""
        try:
            self.logger.info("使用混合模式处理")

            # 使用 Qwen2.5 生成简洁回复
            qwen_response = ""
            for chunk in self.qwen_model.generate(user_input, conversation_history):
                qwen_response += chunk

            # 使用 DeepSeek 生成补充信息
            deepseek_prompt = f"请对以下问题提供专业的补充说明：{user_input}\n原始回答：{qwen_response}"
            deepseek_response = ""
            for chunk in self.deepseek_model.generate(deepseek_prompt, conversation_history):
                deepseek_response += chunk

            # 组合两个模型的回复
            if len(qwen_response.strip()) > 0 and len(deepseek_response.strip()) > 0:
                combined_response = f"综合回复：\n\n{qwen_response}\n\n补充信息：\n{deepseek_response}"
                
                # 流式返回组合后的回复
                for char in combined_response:
                    yield char
            else:
                # 如果任一模型失败，使用成功的那个模型的回复
                response = qwen_response if len(qwen_response.strip()) > 0 else deepseek_response
                for char in response:
                    yield char

        except Exception as e:
            self.logger.error(f"混合模式处理失败: {str(e)}")
            yield f"抱歉，混合模式处理失败: {str(e)}"
    
    @retry(max_attempts=3, delay=1.0)
    def _call_model_api(self, model_name: str, prompt: str, params: Dict[str, Any] = None) -> str:
        """
        调用模型API的通用方法，带重试机制
        
        Args:
            model_name: 模型名称
            prompt: 提示文本
            params: 模型参数
            
        Returns:
            模型生成的回复
        """
        # 这里会在实现具体模型API调用后实现
        # 目前返回模拟数据
        self.logger.info(f"调用{model_name} API")
        time.sleep(1)  # 模拟API调用延迟
        return f"{model_name}生成的回复"