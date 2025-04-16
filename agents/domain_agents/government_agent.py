import json
import time
from typing import Dict, Any, List, Optional, Union, Iterator

from utils.logger import get_logger
from utils.helper_functions import retry, extract_keywords
from agents.sentiment_agent import SentimentAgent

# 导入模型
from models.qwen_model import Qwen2Model
from models.deepseek_model import DeepSeekModel

class GovernmentAgent:
    """
    政务服务Agent，负责处理政务相关的查询和服务
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化政务服务Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("government_agent")
        self.logger.info("政务服务Agent初始化")
        self.sentiment_agent = SentimentAgent(config)
        
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
        
        # 政务服务知识库（简化版）
        self.knowledge_base = {
            "证件办理": {
                "身份证": ["身份证办理", "身份证更换", "身份证挂失", "临时身份证"],
                "护照": ["护照办理", "护照更新", "护照签证", "护照挂失"],
                "驾驶证": ["驾驶证考试", "驾驶证更换", "驾驶证年审", "驾驶证补办"]
            },
            "社会保障": {
                "医疗保险": ["医保报销", "医保缴费", "医保卡办理", "异地就医"],
                "养老保险": ["养老金领取", "养老保险缴费", "退休办理", "养老金计算"],
                "失业保险": ["失业金申领", "失业登记", "再就业培训", "失业保险缴费"]
            },
            "住房服务": {
                "公积金": ["公积金查询", "公积金提取", "公积金贷款", "公积金缴存"],
                "保障房": ["保障房申请", "廉租房", "经济适用房", "公租房"],
                "不动产登记": ["房产证办理", "不动产权证", "房屋过户", "抵押登记"]
            },
            "税务服务": {
                "个人所得税": ["个税申报", "个税计算", "个税退税", "专项附加扣除"],
                "增值税": ["增值税申报", "增值税发票", "增值税退税", "小规模纳税人"],
                "企业所得税": ["企业所得税申报", "企业所得税优惠", "企业所得税计算", "企业所得税减免"]
            },
            "出行服务": {
                "交通违章": ["违章查询", "违章处理", "罚款缴纳", "交通违章申诉"],
                "公共交通": ["公交卡办理", "地铁乘车码", "公共自行车", "老年卡办理"],
                "机动车": ["车辆年检", "车辆过户", "车牌摇号", "机动车报废"]
            }
        }
        
        # 政务服务指南
        self.service_guides = {
            "身份证办理": "亲爱的市民朋友，关于身份证办理，我来为您详细介绍：\n\n【办理流程】\n1. 准备材料\n   - 户口本原件\n   - 旧身份证（如有）\n   - 近期免冠照片（也可现场拍摄）\n\n2. 办理地点和方式\n   - 就近选择户籍所在地派出所或户籍办理点\n   - 建议提前通过互联网进行预约\n\n3. 具体步骤\n   - 到达现场后先取号\n   - 填写《居民身份证申领登记表》\n   - 验证身份信息\n   - 采集照片（如需）\n   - 缴纳工本费（首次办理免费）\n\n4. 领取方式\n   - 一般15-30天制作完成\n   - 可选择现场领取或邮寄到家\n\n【温馨提示】\n✦ 首次申领免收工本费，丢失补办需缴费\n✦ 照片可现场拍摄或自带（需符合规格要求）\n✦ 建议避开工作日高峰时段办理\n✦ 可通过'全国公安政务服务平台'预约办理\n✦ 紧急情况可申请加急办理（可能需要额外费用）\n\n如果您在办理过程中遇到任何问题，随时可以询问我！",

            "医保报销": "亲爱的参保人，关于医保报销事项，我来为您详细说明：\n\n【报销材料准备】\n1. 必需材料\n   - 医保卡\n   - 有效身份证件\n   - 医疗费用票据原件\n   - 病历本或出院小结\n   - 处方单据\n   - 检查化验报告单\n\n2. 报销途径选择\n   A. 线下报销\n      - 前往医保经办机构\n      - 社区服务中心\n      - 定点医院医保窗口\n   B. 线上报销\n      - 医保APP\n      - 各地医保网上服务平台\n\n3. 报销流程\n   - 材料准备与审核\n   - 填写报销申请表\n   - 提交材料\n   - 等待审核\n   - 资金到账（一般5-15个工作日）\n\n【温馨提示】\n✦ 及时报销，发票超过3个月可能无法受理\n✦ 保管好所有原始单据\n✦ 大额医疗费用建议当面办理\n✦ 可通过医保APP实时查询报销进度\n✦ 异地就医先备案，报销更便捷\n\n如有任何疑问，我很乐意为您解答！",

            "公积金提取": "亲爱的缴存职工，关于公积金提取，我来为您详细介绍：\n\n【提取条件】\n1. 购房提取\n   - 购买自住住房\n   - 偿还住房贷款\n   - 支付房租\n\n2. 其他提取情形\n   - 离职后提取\n   - 退休提取\n   - 大病医疗提取\n   - 本人死亡或完全丧失劳动能力\n\n【办理流程】\n1. 准备材料\n   - 身份证原件\n   - 公积金联名卡\n   - 提取证明材料（如购房合同、租赁合同等）\n\n2. 办理方式\n   A. 线上办理（推荐）\n      - 公积金APP\n      - 公积金网上服务大厅\n   B. 线下办理\n      - 公积金管理中心\n      - 授权银行网点\n\n3. 具体步骤\n   - 选择提取类型\n   - 提交申请材料\n   - 等待审核\n   - 资金到账（一般1-3个工作日）\n\n【温馨提示】\n✦ 提前了解提取条件和额度限制\n✦ 准备完整的证明材料，避免多次往返\n✦ 可通过APP预约办理，避免排队\n✦ 部分业务支持'刷脸'办理\n✦ 提取后建议查询账户变动情况\n\n如果您在办理过程中有任何疑问，随时可以询问我哦！",

            "个税申报": "亲爱的纳税人，关于个人所得税申报，我来为您详细说明：\n\n【申报时间】\n- 每月1日至15日进行上月收入申报\n- 每年3-6月份进行年度汇算\n\n【申报渠道】\n1. 手机端\n   - 个人所得税APP（推荐）\n   - 微信小程序\n\n2. 电脑端\n   - 自然人电子税务局网站\n   - 各地税务局网上办税平台\n\n【申报流程】\n1. 登录系统\n   - 注册个人所得税账号\n   - 实名认证\n\n2. 信息确认\n   - 核对收入信息\n   - 确认专项附加扣除\n   - 补充其他收入信息\n\n3. 提交申报\n   - 系统自动计算应纳税额\n   - 确认无误后提交\n   - 如有退税，等待退税到账\n\n【温馨提示】\n✦ 及时更新个人信息和专项附加扣除信息\n✦ 妥善保管发票等税收凭证\n✦ 设置申报提醒，避免超期\n✦ 遇到问题可拨打12366咨询\n✦ 注意保护个人税收信息安全\n\n如果您在申报过程中遇到任何问题，我都可以为您解答！",

            "违章处理": "亲爱的车主，关于交通违章处理，我来为您详细介绍：\n\n【查询方式】\n1. 线上查询\n   - 交管12123APP（推荐）\n   - 全国交通安全综合服务平台\n   - 各地交管网站\n\n2. 线下查询\n   - 交警大队\n   - 车管所\n   - 违章处理点\n\n【处理流程】\n1. 线上处理（适用于部分轻微违章）\n   - 登录12123APP\n   - 查询违章记录\n   - 选择需处理的违章\n   - 在线缴纳罚款\n   - 扣分自动处理\n\n2. 线下处理\n   - 前往指定地点\n   - 提供车辆信息\n   - 缴纳罚款\n   - 处理扣分\n\n【所需材料】\n- 驾驶证\n- 行驶证\n- 车主身份证\n- 违章通知书（如有）\n\n【温馨提示】\n✦ 及时处理违章，避免影响年检\n✦ 注意违章处理期限\n✦ 可设置违章提醒服务\n✦ 累积记分周期为12个月\n✦ 某些违章可能需要现场处理\n\n如果您在处理过程中有任何疑问，随时可以询问我！",

            "护照办理": "亲爱的市民朋友，关于护照办理，我来为您详细介绍：\n\n【办理条件】\n- 年满16周岁可独立办理\n- 未满16周岁需监护人陪同\n- 身份信息真实有效\n\n【准备材料】\n1. 基本材料\n   - 身份证原件\n   - 户口本原件\n   - 近期证件照片\n\n2. 特殊情况补充材料\n   - 未成年人需提供监护人身份证明\n   - 加急办理需提供证明材料\n\n【办理流程】\n1. 预约\n   - 网上预约（推荐）\n   - 现场取号\n\n2. 现场办理\n   - 资料审核\n   - 照片采集\n   - 面谈确认\n   - 缴费\n\n3. 领取\n   - 一般7-10个工作日\n   - 可选择现场领取或邮寄\n\n【温馨提示】\n✦ 照片需符合护照照片要求\n✦ 建议提前网上预约，避免排队\n✦ 可办理加急服务（额外收费）\n✦ 注意护照有效期（一般10年）\n✦ 建议预留充足办理时间\n\n如果您在办理过程中遇到任何问题，随时可以询问我！",

            "养老金领取": "亲爱的退休人员，关于养老金领取，我来为您详细说明：\n\n【领取条件】\n1. 基本条件\n   - 达到法定退休年龄\n   - 缴费年限符合规定\n   - 办理退休手续\n\n2. 特殊情况\n   - 提前退休\n   - 特殊工种退休\n   - 病退\n\n【办理材料】\n- 身份证原件\n- 退休证\n- 社保卡\n- 银行卡\n- 照片\n- 退休审批表\n\n【办理流程】\n1. 提交申请\n   - 前往社保经办机构\n   - 填写领取申请表\n   - 提供相关材料\n\n2. 信息确认\n   - 核实个人信息\n   - 确认待遇领取方式\n   - 选择发放账户\n\n3. 待遇发放\n   - 首次发放一般在1-2个月内\n   - 后续按月发放\n\n【温馨提示】\n✦ 确保社保缴费记录完整\n✦ 可选择银行代发或社保卡领取\n✦ 注意及时更新个人信息\n✦ 定期领取待遇资格认证\n✦ 如有变动及时报告\n\n如果您在领取过程中有任何疑问，我都可以为您解答！"} # 服务指南字典结束
        
        # 添加服务类型权重
        self.service_weights = {
            "证件办理": 1.0,
            "社会保障": 1.0,
            "住房服务": 0.9,
            "税务服务": 0.9,
            "出行服务": 0.8
        }
        
        self.response_times = []
    
    def _build_prompt(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """构建提示词
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            构建的提示词
        """
        # 提取关键词
        keywords = extract_keywords(user_input)
        
        # 构建提示词
        prompt = f"作为一个专业的政务服务助手，请帮助解答以下问题:\n{user_input}\n"
        
        # 添加知识库相关内容
        for service_type, services in self.knowledge_base.items():
            if any(kw in str(services) for kw in keywords):
                prompt += f"\n参考{service_type}相关服务："
                for service, details in services.items():
                    if any(kw in str(details) for kw in keywords):
                        prompt += f"\n- {service}: {', '.join(details)}"
                        # 添加详细服务指南
                        if service in self.service_guides:
                            prompt += f"\n\n服务指南:\n{self.service_guides[service]}"
        
        return prompt
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理政务相关的查询
        
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
                "service_type": "",
                "query_type": "",
                "response": "",
                "processing_time": 0,
                "sentiment": None
            }
            
            if context is None:
                context = {}
            
            self.logger.info(f"处理政务查询: {user_input}")
            
            # 情感分析
            sentiment_result = self.sentiment_agent.process(user_input)
            result["sentiment"] = sentiment_result["sentiment"]
            self.logger.debug(f"情感分析结果: {sentiment_result['sentiment']}")
            
            # 提取关键词
            keywords = extract_keywords(user_input)
            self.logger.debug(f"提取的关键词: {keywords}")
            
            # 分析查询类型
            query_type, service_category = self._analyze_query(user_input, keywords)
            result["query_type"] = query_type
            result["service_type"] = service_category
            
            # 特殊处理身份证办理相关查询
            if "身份证" in user_input and any(word in user_input for word in ["办理", "更换", "换", "到期"]):
                response = self.service_guides.get("身份证办理", "")
                if response:
                    # 根据情感调整回复语气
                    response = self._adjust_response_tone(response, sentiment_result['sentiment'])
                    result["response"] = response
                    result["processing_time"] = time.time() - start_time
                    self.response_times.append(result["processing_time"])
                    return result
            
            # 构建提示词
            prompt = self._build_prompt(user_input, query_type, service_category, keywords)
            
            # 获取对话历史
            conversation_history = context.get("conversation_history", [])
            
            # 使用模型生成回复
            model_strategy = context.get("model_strategy", "自动（智能选择）")
            if model_strategy in self.model_selection_strategies:
                response = "".join(chunk for chunk in self.model_selection_strategies[model_strategy](prompt, conversation_history))
            else:
                response = "".join(chunk for chunk in self._auto_select_model(prompt, conversation_history))
            
            # 根据情感调整回复语气
            response = self._adjust_response_tone(response, sentiment_result['sentiment'])
            result["response"] = response
            
            result["processing_time"] = time.time() - start_time
            self.response_times.append(result["processing_time"])
            return result
            
        except Exception as e:
            self.logger.error(f"处理政务查询失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    def _build_prompt(self, user_input: str, query_type: str, service_category: str, keywords: List[str]) -> str:
        """
        构建模型提示词
        """
        # 基础提示词
        prompt = f"作为政务服务人员，请回答以下问题：\n{user_input}\n\n"
        
        # 根据查询类型添加特定提示
        if query_type == "service_info":
            if service_category in self.knowledge_base:
                prompt += f"这是关于{service_category}的咨询，请提供相关服务信息。\n"
                if keywords:
                    prompt += f"重点关注这些方面：{', '.join(keywords)}\n"
        elif query_type == "procedure_guide":
            prompt += "请详细说明办理流程和所需材料。\n"
        elif query_type == "policy_query":
            prompt += "请解释相关政策规定和要求。\n"
        elif query_type == "location_query":
            prompt += "请提供相关服务网点和办理地点信息。\n"
        
        # 添加服务态度要求
        prompt += "\n请以专业、耐心、友善的态度回答，确保信息准确完整。"
        
        return prompt
    
    def _adjust_response_tone(self, response: str, sentiment: str) -> str:
        """
        根据用户情感调整回复语气
        
        Args:
            response: 原始回复
            sentiment: 情感分析结果
            
        Returns:
            调整后的回复
        """
        if sentiment == "negative":
            # 对于负面情绪，增加安慰和鼓励
            prefix = "我理解您的心情，让我来帮您解决这个问题。\n"
            suffix = "\n\n如果您还有任何疑问，随时都可以询问我。我会尽最大努力为您提供帮助。"
        elif sentiment == "positive":
            # 对于正面情绪，表达赞同和支持
            prefix = "很高兴看到您对这件事这么有热情！\n"
            suffix = "\n\n希望这些信息对您有帮助。祝您办事顺利！"
        else:
            # 对于中性情绪，保持专业客观
            prefix = ""
            suffix = "\n\n如果您需要更多信息，请随时询问。"
        
        return prefix + response + suffix
        
    def _auto_select_model(self, user_input: str, conversation_history: List[Dict[str, str]] = None) -> Union[str, Iterator[str]]:
        """
        智能选择合适的模型处理用户输入
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            
        Returns:
            模型生成的回复
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
            deepseek_prompt = f"请以政务服务人员的身份回答以下问题：\n{user_input}"
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
        
    def _analyze_query(self, query: str, keywords: List[str]) -> tuple:
            """
            分析查询类型和服务类别
            
            Args:
                query: 用户查询
                keywords: 提取的关键词
                
            Returns:
                (查询类型, 服务类别) 元组
            """
            # 识别服务类别
            service_category = None
            for category in self.knowledge_base.keys():
                if category in query or any(keyword in category for keyword in keywords):
                    service_category = category
                    break
            
            if not service_category:
                # 尝试从子类别中匹配
                for category, subcategories in self.knowledge_base.items():
                    for subcategory in subcategories.keys():
                        if subcategory in query or any(keyword in subcategory for keyword in keywords):
                            service_category = category
                            break
                    if service_category:
                        break
            
            # 识别查询类型
            if "怎么办" in query or "如何办理" in query or "流程" in query or "步骤" in query:
                return "procedure_guide", service_category
            elif "在哪里" in query or "地点" in query or "地址" in query or "哪儿" in query:
                return "location_query", service_category
            elif "政策" in query or "规定" in query or "法规" in query or "条例" in query:
                return "policy_query", service_category
            elif service_category:
                return "service_info", service_category
            else:
                return "general_query", None
        
    def _provide_service_info(self, service_category: str, keywords: List[str]) -> str:
            """
            提供政务服务信息
            
            Args:
                service_category: 服务类别
                keywords: 关键词列表
                
            Returns:
                服务信息
            """
            try:
                if not service_category:
                    return (
                        "尊敬的市民，您好！\n\n"
                        "我是您的智能政务服务助手，可以为您提供以下服务类别的咨询：\n\n"
                        "1. 证件办理：身份证、护照、驾驶证等\n"
                        "2. 社会保障：医保、养老保险、失业保险等\n"
                        "3. 住房服务：公积金、保障房、不动产登记等\n"
                        "4. 税务服务：个税、增值税、企业所得税等\n"
                        "5. 出行服务：交通违章、公共交通、机动车等\n\n"
                        "请告诉我您需要了解哪方面的具体服务，我会为您提供专业的指导。"
                    )
                
                # 查找具体服务
                specific_service = None
                specific_service_info = None
                service_matches = []
                
                # 增强服务匹配逻辑
                for subcategory, services in self.knowledge_base.get(service_category, {}).items():
                    for service in services:
                        match_score = 0
                        # 计算关键词匹配度
                        for keyword in keywords:
                            if keyword in service:
                                match_score += 1
                        if match_score > 0:
                            service_matches.append((service, match_score))
                
                # 按匹配度排序
                if service_matches:
                    service_matches.sort(key=lambda x: x[1], reverse=True)
                    specific_service = service_matches[0][0]
                    specific_service_info = self.service_guides.get(specific_service)
                
                if specific_service and specific_service_info:
                    # 添加服务分类标签
                    service_tags = self._get_service_tags(specific_service)
                    return (
                        f"关于【{specific_service}】的服务指南\n"
                        f"服务分类：{service_tags}\n\n"
                        f"{specific_service_info}\n\n"
                        f"温馨提示：如需了解更多详情或有其他问题，请随时询问。"
                    )
                elif service_category:
                    subcategories = list(self.knowledge_base[service_category].keys())
                    response = f"在【{service_category}】类别下，我们提供以下具体服务：\n\n"
                    for i, subcat in enumerate(subcategories, 1):
                        services = self.knowledge_base[service_category][subcat]
                        response += f"{i}. {subcat}：\n"
                        for service in services:
                            response += f"   - {service}\n"
                    response += "\n请问您想了解哪项具体服务的办理指南？"
                    return response
                else:
                    return self._general_government_response("")
                    
            except Exception as e:
                self.logger.error(f"提供服务信息失败: {str(e)}")
                return "抱歉，获取服务信息时遇到问题。请重新描述您需要了解的服务，我会尽力为您提供帮助。"
                
    def _get_service_tags(self, service: str) -> str:
            """获取服务标签"""
            tags = []
            
            # 紧急程度
            if any(word in service for word in ["挂失", "补办", "临时"]):
                tags.append("紧急办理")
            
            # 办理方式
            if "网上" in service or "在线" in service:
                tags.append("可在线办理")
            elif "现场" in service:
                tags.append("需现场办理")
            
            # 服务对象
            if "个人" in service:
                tags.append("个人业务")
            elif "企业" in service:
                tags.append("企业业务")
            
            # 默认标签
            if not tags:
                tags.append("标准服务")
                
            return "、".join(tags)
        
    def _provide_procedure_guide(self, query: str, keywords: List[str]) -> str:
            """
            提供办事流程指南
            
            Args:
                query: 用户查询
                keywords: 关键词列表
                
            Returns:
                流程指南
            """
            # 查找匹配的服务指南
            for service, guide in self.service_guides.items():
                if service in query or any(keyword in service for keyword in keywords):
                    return f"{service}的办理流程：\n\n{guide}"
            
            # 如果没有找到具体服务，提供一般性回复
            return "您想了解哪项具体服务的办理流程？我可以提供身份证办理、医保报销、公积金提取、个税申报和违章处理等多项服务的详细流程指南。"
        
    def _provide_policy_info(self, service_category: str, keywords: List[str]) -> str:
            """
            提供政策信息
            
            Args:
                service_category: 服务类别
                keywords: 关键词列表
                
            Returns:
                政策信息
            """
            # 这里应该连接到政策数据库或知识库
            # 简化版实现，返回一般性政策信息
            if service_category == "证件办理":
                return "证件办理相关政策：根据《中华人民共和国居民身份证法》，公民应当依法申领居民身份证。首次申领居民身份证不收取工本费，换领、补领居民身份证应当缴纳工本费。"
            elif service_category == "社会保障":
                return "社会保障相关政策：根据《社会保险法》，用人单位应当为其职工缴纳基本养老保险、基本医疗保险、工伤保险、失业保险和生育保险费用。个人应当缴纳基本养老保险和基本医疗保险费用。"
            elif service_category == "住房服务":
                return "住房服务相关政策：根据《住房公积金管理条例》，单位和职工个人缴存的住房公积金，属于职工个人所有。职工有权按照规定提取本人住房公积金账户内的存储余额，用于购买、建造、翻建、大修自住住房等。"
            elif service_category == "税务服务":
                return "税务服务相关政策：根据《个人所得税法》，居民个人取得综合所得，按年计算个人所得税；非居民个人取得综合所得，按月或者按次计算个人所得税。纳税人可以享受专项附加扣除。"
            elif service_category == "出行服务":
                return "出行服务相关政策：根据《道路交通安全法》，机动车驾驶人应当按照规定定期参加审验。机动车应当依法进行登记，并按照规定检验合格后，方可上道路行驶。"
            else:
                return "您想了解哪方面的政策信息？我可以提供证件办理、社会保障、住房服务、税务服务和出行服务等多个领域的政策解读。"
        
    def _provide_service_location(self, service_category: str) -> str:
            """
            提供服务办理地点信息
            
            Args:
                service_category: 服务类别
                
            Returns:
                地点信息
            """
            # 这里应该连接到地点数据库
            # 简化版实现，返回一般性地点信息
            if service_category == "证件办理":
                return "证件办理地点：身份证可在户籍所在地派出所办理；护照可在出入境管理局办理；驾驶证可在车管所办理。具体地址请查询当地政府网站或拨打政务服务热线。"
            elif service_category == "社会保障":
                return "社会保障服务地点：医疗保险、养老保险和失业保险业务可在当地社保局或社保服务中心办理。许多服务也可通过线上平台如‘掌上社保’APP办理。"
            elif service_category == "住房服务":
                return "住房服务地点：公积金业务可在住房公积金管理中心办理；保障房申请可在住房保障部门办理；不动产登记可在不动产登记中心办理。"
            elif service_category == "税务服务":
                return "税务服务地点：个人所得税、增值税和企业所得税业务可在当地税务局办理。许多税务服务也可通过电子税务局网站或个人所得税APP办理。"
            elif service_category == "出行服务":
                return "出行服务地点：交通违章处理可在交通管理部门或通过交管12123APP办理；公共交通卡可在指定的服务网点办理；机动车业务可在车管所办理。"
            else:
                return "不同的政务服务有不同的办理地点。请问您具体想了解哪类服务的办理地点？我可以提供证件办理、社会保障、住房服务、税务服务和出行服务等多个领域的办理地点信息。"
        
    def _general_government_response(self, query: str) -> str:
            """
            提供一般性政务回复
            
            Args:
                query: 用户查询
                
            Returns:
                一般性回复
            """
            return "您好，我是政务服务助手，可以为您提供证件办理、社会保障、住房服务、税务服务和出行服务等多方面的政务信息和指导。请问您需要了解哪方面的具体服务？"
        
    def get_service_categories(self) -> List[str]:
            """
            获取支持的服务类别
            
            Returns:
                服务类别列表
            """
            return list(self.knowledge_base.keys())