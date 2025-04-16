import json
import time
from typing import Dict, Any, List, Optional, Union, Iterator

from utils.logger import get_logger
from utils.helper_functions import retry, extract_keywords

# 导入模型
from models.qwen_model import Qwen2Model
from models.deepseek_model import DeepSeekModel

class EducationAgent:
    """
    教育Agent，负责处理教育相关的查询和辅导
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化教育Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("education_agent")
        self.logger.info("教育Agent初始化")
        
        # 教育领域知识库（简化版）
        self.knowledge_base = {
            "数学": {
                "代数": ["方程式", "函数", "多项式", "矩阵"],
                "几何": ["三角形", "圆", "椭圆", "向量"],
                "微积分": ["导数", "积分", "极限", "微分方程"]
            },
            "物理": {
                "力学": ["牛顿定律", "动量", "能量守恒"],
                "电磁学": ["电场", "磁场", "电磁波"],
                "热力学": ["熵", "热力学定律", "热传导"]
            },
            "化学": {
                "有机化学": ["烃类", "醇", "酸"],
                "无机化学": ["元素周期表", "化学键", "氧化还原"],
                "物理化学": ["化学平衡", "反应动力学", "热化学"]
            },
            "语文": {
                "古代文学": ["诗词", "散文", "小说"],
                "现代文学": ["小说", "散文", "戏剧"],
                "语法修辞": ["修辞手法", "句法分析", "词汇"]
            },
            "英语": {
                "语法": ["时态", "语态", "从句"],
                "词汇": ["同义词", "反义词", "词根词缀"],
                "阅读": ["理解", "推断", "主旨"]
            }
        }
        
        # 添加学科权重配置
        self.subject_weights = {
            "数学": 1.0,
            "物理": 1.0,
            "化学": 1.0,
            "语文": 0.9,
            "英语": 0.9
        }
        
        # 添加查询类型权重
        self.query_weights = {
            "概念": 0.9,
            "题目": 1.0,
            "例题": 0.8,
            "练习": 0.7,
            "考试": 0.8
        }
        
        # 初始化模型
        self.qwen_model = Qwen2Model(config["models"]["qwen"])
        self.deepseek_model = DeepSeekModel(config["models"]["deepseek"])
        
        # 模型选择策略
        self.model_selection_strategies = {
            "自动（智能选择）": self._auto_select_model,
            "Qwen2.5": self._use_qwen_model,
            "DeepSeek": self._use_deepseek_model,
            "混合模式": self._use_hybrid_model
        }
        
        self.response_times = []
    
    def _build_prompt(self, user_input: str, query_type: str = None, subject: str = None, keywords: List[str] = None) -> str:
        """构建提示词
        
        Args:
            user_input: 用户输入文本
            query_type: 查询类型
            subject: 学科类型
            keywords: 关键词列表
            
        Returns:
            构建的提示词
        """
        # 如果没有提供关键词，则提取关键词
        if keywords is None:
            keywords = extract_keywords(user_input)
        
        # 构建基础提示词
        prompt = f"作为一个专业的教育辅导助手，请帮助解答以下问题:\n{user_input}\n"
        
        # 添加查询类型相关提示
        if query_type == "subject_info":
            prompt += "\n请详细解释相关概念，包括定义、特点和应用场景。"
        elif query_type == "problem_solving":
            prompt += "\n请提供详细的解题思路和步骤。"
        elif query_type == "resource_recommendation":
            prompt += "\n请推荐相关的学习资源和参考材料。"
        
        # 添加知识库相关内容
        if subject and subject in self.knowledge_base:
            prompt += f"\n参考{subject}相关知识："
            for topic, subtopics in self.knowledge_base[subject].items():
                if any(kw in str(subtopics) for kw in keywords):
                    prompt += f"\n- {topic}: {', '.join(subtopics)}"
        else:
            # 如果没有指定学科，搜索所有知识库
            for subject, topics in self.knowledge_base.items():
                if any(kw in str(topics) for kw in keywords):
                    prompt += f"\n参考{subject}相关知识："
                    for topic, subtopics in topics.items():
                        if any(kw in str(subtopics) for kw in keywords):
                            prompt += f"\n- {topic}: {', '.join(subtopics)}"
        
        return prompt

    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理教育相关的查询
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            处理结果
        """
        start_time = time.time()
        try:
            result = {
                "success": True,
                "subject": "",
                "query_type": "",
                "response": "",
                "processing_time": 0
            }
            
            if context is None:
                context = {}
            
            self.logger.info(f"处理教育查询: {user_input}")
            
            # 提取关键词
            keywords = extract_keywords(user_input)
            self.logger.debug(f"提取的关键词: {keywords}")
            
            # 分析查询类型
            query_type, subject = self._analyze_query(user_input, keywords)
            result["subject"] = subject
            result["query_type"] = query_type
            
            # 构建提示词
            prompt = self._build_prompt(user_input, query_type, subject, keywords)
            
            # 获取对话历史
            conversation_history = context.get("conversation_history", [])
            
            # 使用模型生成回复
            model_strategy = context.get("model_strategy", "自动（智能选择）")
            if model_strategy in self.model_selection_strategies:
                response = self.model_selection_strategies[model_strategy](prompt, conversation_history)
            else:
                response = self._auto_select_model(prompt, conversation_history)
                
            result["response"] = response
            
            result["processing_time"] = time.time() - start_time
            self.response_times.append(result["processing_time"])
            return result
            
        except Exception as e:
            self.logger.error(f"处理教育查询失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _analyze_query(self, query: str, keywords: List[str]) -> tuple:
        """
        分析查询类型和学科
        
        Args:
            query: 用户查询
            keywords: 提取的关键词
            
        Returns:
            (查询类型, 学科) 元组
        """
        scores = {subject: 0.0 for subject in self.subject_weights}
        for keyword in keywords:
            for subject, weight in self.subject_weights.items():
                if keyword in self.knowledge_base.get(subject, {}):
                    scores[subject] += weight
                    
        best_subject = max(scores.items(), key=lambda x: x[1])[0] if any(scores.values()) else "通用"

        # 识别查询类型
        if "什么是" in query or "概念" in query or "定义" in query or "介绍" in query:
            query_type = "subject_info"
        elif "问题" in query or "解答" in query or "怎么做" in query or "如何解" in query:
            query_type = "problem_solving"
        elif "资源" in query or "教材" in query or "书籍" in query or "视频" in query or "推荐" in query:
            query_type = "learning_resource"
        else:
            query_type = "general"

        return query_type, best_subject
    
    def _provide_subject_info(self, subject: str) -> str:
        """
        提供学科信息
        
        Args:
            subject: 学科名称
            
        Returns:
            学科信息
        """
        if subject in self.knowledge_base:
            domains = self.knowledge_base[subject].keys()
            response = f"关于{subject}学科，它主要包含以下领域：\n\n"
            for domain in domains:
                topics = self.knowledge_base[subject][domain]
                response += f"- {domain}：{', '.join(topics)}\n"
            
            response += f"\n您对{subject}的哪个具体领域感兴趣？我可以提供更详细的信息。"
            return response
        else:
            return f"抱歉，我目前没有关于{subject}的详细信息。您可以询问数学、物理、化学、语文或英语等学科的内容。"
    
    def _solve_problem(self, query: str, subject: str) -> str:
        """
        解答问题
        
        Args:
            query: 用户查询
            subject: 学科名称
            
        Returns:
            问题解答
        """
        try:
            # 根据学科和关键词匹配最佳解答策略
            if subject == "数学":
                return self._solve_math_problem(query)
            elif subject == "物理":
                return self._solve_physics_problem(query)
            elif subject == "化学":
                return self._solve_chemistry_problem(query)
            elif subject == "语文":
                return self._solve_chinese_problem(query)
            elif subject == "英语":
                return self._solve_english_problem(query)
            else:
                return "抱歉，我暂时无法解答这个问题。请尝试更具体地描述您的问题，或者询问其他学科的内容。"
                
        except Exception as e:
            self.logger.error(f"解答问题失败: {str(e)}")
            return "抱歉，解答问题时遇到了错误。请重新描述您的问题，我会尽力帮助您。"
    
    def _solve_math_problem(self, query: str) -> str:
        """解答数学问题"""
        if "方程" in query:
            return (
                "解方程的详细步骤：\n\n"
                "1. 预处理阶段\n"
                "   - 仔细阅读题目，明确未知数\n"
                "   - 检查方程是否有分数或根式\n\n"
                "2. 化简阶段\n"
                "   - 去分母：通分化整\n"
                "   - 去括号：分配律展开\n"
                "   - 合并同类项\n\n"
                "3. 求解阶段\n"
                "   - 移项：变号移项\n"
                "   - 系数化一：求解未知数\n"
                "   - 验证：代入原方程检查\n\n"
                "示例：2x + 3 = 7\n"
                "1) 移项：2x = 7 - 3\n"
                "2) 化简：2x = 4\n"
                "3) 求解：x = 2\n"
                "4) 验证：2(2) + 3 = 7 ✓\n\n"
                "提示：解方程时要注意：\n"
                "- 始终保持等式两边相等\n"
                "- 记录每一步的运算过程\n"
                "- 最后验证答案"
            )
        elif "函数" in query:
            return (
                "函数的核心概念与应用：\n\n"
                "1. 基本定义\n"
                "   函数是描述两个集合之间对应关系的数学概念：\n"
                "   - 定义域：自变量x的取值范围\n"
                "   - 值域：因变量y的取值范围\n"
                "   - 对应关系：每个x唯一对应一个y\n\n"
                "2. 常见函数类型\n"
                "   a) 线性函数: f(x) = ax + b\n"
                "      - 图像是直线\n"
                "      - a决定斜率，b决定截距\n\n"
                "   b) 二次函数: f(x) = ax² + bx + c\n"
                "      - 图像是抛物线\n"
                "      - a决定开口方向和宽窄\n"
                "      - 对称轴：x = -b/(2a)\n\n"
                "   c) 指数函数: f(x) = aˣ (a>0且a≠1)\n"
                "      - 图像经过点(0,1)\n"
                "      - a>1时单调递增\n"
                "      - 0<a<1时单调递减\n\n"
                "   d) 对数函数: f(x) = logₐx\n"
                "      - 是指数函数的反函数\n"
                "      - 定义域是正实数\n"
                "      - 图像经过点(1,0)\n\n"
                "3. 应用场景\n"
                "   - 线性函数：成本分析、距离-时间关系\n"
                "   - 二次函数：抛物运动、最优化问题\n"
                "   - 指数函数：人口增长、复利计算\n"
                "   - 对数函数：地震强度、pH值计算"
            )
        elif "三角" in query:
            return (
                "三角函数与三角恒等式：\n\n"
                "1. 基本三角函数\n"
                "   - 正弦：sin θ = 对边/斜边\n"
                "   - 余弦：cos θ = 邻边/斜边\n"
                "   - 正切：tan θ = 对边/邻边\n\n"
                "2. 重要角度值\n"
                "   0°: (1, 0, 0)\n"
                "   30°: (1/2, √3/2, 1/√3)\n"
                "   45°: (√2/2, √2/2, 1)\n"
                "   60°: (√3/2, 1/2, √3)\n"
                "   90°: (0, 1, 不存在)\n\n"
                "3. 基本恒等式\n"
                "   - sin²θ + cos²θ = 1\n"
                "   - tan θ = sin θ / cos θ\n"
                "   - sin(A±B) = sinA·cosB ± cosA·sinB"
            )
        else:
            return "请具体说明您想了解的数学概念或问题类型，例如：方程、函数、三角函数等。我会为您提供详细的讲解和示例。"
        
        return f"您的问题涉及{subject}领域。要解答这类问题，建议先明确概念，然后应用相关公式或方法。如果您能提供具体的问题，我可以给出更详细的解答。"
    
    def _recommend_resources(self, subject: str) -> str:
        """
        推荐学习资源
        
        Args:
            subject: 学科名称
            
        Returns:
            资源推荐
        """
        resources = {
            "数学": [
                "《数学分析》 - 陈纪修、於崇华、金路",
                "《高等代数》 - 北京大学数学系",
                "可汗学院 (Khan Academy) - 免费数学视频教程",
                "3Blue1Brown - YouTube数学可视化频道"
            ],
            "物理": [
                "《费曼物理学讲义》 - 理查德·费曼",
                "《大学物理学》 - 赵凯华、陈熙谋",
                "MIT开放课程 - 物理系列",
                "PhET互动模拟 - 物理实验模拟平台"
            ],
            "化学": [
                "《普通化学原理》 - 华彤文等",
                "《有机化学》 - 胡宏纹",
                "化学之美 - 科普网站",
                "Royal Society of Chemistry - 化学资源网站"
            ],
            "语文": [
                "《古代汉语》 - 王力",
                "《文学理论教程》 - 童庆炳",
                "中国诗词大会 - 电视节目",
                "古诗文网 - 古代文学资源库"
            ],
            "英语": [
                "《新概念英语》系列",
                "《剑桥英语语法》 - Raymond Murphy",
                "BBC Learning English - 英语学习网站",
                "TED Talks - 英语演讲视频"
            ],
            "通用": [
                "中国大学MOOC - 多学科在线课程平台",
                "学堂在线 - 清华大学创办的MOOC平台",
                "Coursera - 国际知名在线教育平台",
                "网易公开课 - 多领域视频教程"
            ]
        }
        
        if subject in resources:
            response = f"以下是{subject}学科的推荐学习资源：\n\n"
            for resource in resources[subject]:
                response += f"- {resource}\n"
            
            response += "\n希望这些资源对您的学习有所帮助！如果需要特定领域的资源，请告诉我。"
            return response
        else:
            return self._recommend_resources("通用")
    
    def _general_education_response(self, query: str) -> str:
        """
        通用教育回复
        
        Args:
            query: 用户查询
            
        Returns:
            通用回复
        """
        return f"您的问题是关于教育方面的。我可以提供学科知识、解答问题或推荐学习资源。请告诉我您具体需要哪方面的帮助，例如'数学代数知识'、'物理力学问题'或'英语学习资源推荐'等。"
    
    @retry(max_attempts=2)
    def search_knowledge_base(self, keyword: str) -> List[Dict[str, Any]]:
        """
        搜索知识库（示例方法）
        
        Args:
            keyword: 关键词
            
        Returns:
            搜索结果列表
        """
        # 这里应该实现更复杂的知识库搜索逻辑
        # 目前返回模拟数据
        results = []
        
        for subject, domains in self.knowledge_base.items():
            for domain, topics in domains.items():
                if keyword in domain or keyword in topics:
                    results.append({
                        "subject": subject,
                        "domain": domain,
                        "relevance": 0.85 if keyword in domain else 0.7
                    })
        
        return results

    def _auto_select_model(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Union[str, Iterator[str]]:
        """
        自动选择合适的模型
        
        Args:
            user_input: 用户输入文本
            conversation_history: 对话历史
            
        Returns:
            生成的回复
        """
        # 根据输入特征选择模型
        if len(user_input) > 100 or "详细" in user_input or "解释" in user_input:
            return self._use_deepseek_model(user_input, conversation_history)
        else:
            return self._use_qwen_model(user_input, conversation_history)
    
    def _use_qwen_model(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Union[str, Iterator[str]]:
        """
        使用Qwen2.5模型生成回复
        """
        return self.qwen_model.generate(user_input, conversation_history)
    
    def _use_deepseek_model(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Union[str, Iterator[str]]:
        """
        使用DeepSeek模型生成回复
        """
        messages = self._format_conversation_history(conversation_history) if conversation_history else []
        return self.deepseek_model.generate(user_input, messages)
    
    def _use_hybrid_model(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Union[str, Iterator[str]]:
        """
        混合使用两个模型生成回复
        """
        # 根据输入特征选择是否使用流式生成
        if len(user_input) > 50:
            for chunk in self.qwen_model.generate(user_input, conversation_history):
                yield chunk
        else:
            deepseek_prompt = f"请以教育专家的身份回答以下问题：\n{user_input}"
            for chunk in self.deepseek_model.generate(deepseek_prompt, conversation_history):
                yield chunk
    
    def _format_conversation_history(self, conversation_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        格式化对话历史，适配不同模型的需求
        """
        if not conversation_history:
            return []
        
        formatted_history = []
        for message in conversation_history:
            if "role" in message and "content" in message:
                formatted_history.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        return formatted_history