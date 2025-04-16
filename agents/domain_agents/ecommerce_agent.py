import json
import time
from typing import Dict, Any, List, Optional, Union, Iterator

from utils.logger import get_logger
from utils.helper_functions import retry, extract_keywords

# 修改导入路径为正确的模型路径
from models.qwen_model import Qwen2Model
from models.deepseek_model import DeepSeekModel

class EcommerceAgent:
    """
    电商Agent，负责处理电商相关的查询和服务
    """
    def __init__(self, config: Dict[str, Any]):
        """
        初始化电商Agent
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.logger = get_logger("ecommerce_agent")
        self.logger.info("电商Agent初始化")
        
        # 模拟商品数据库
        self.products_db = {
            "手机": [
                {"id": "p001", "name": "智能手机A", "brand": "品牌X", "price": 2999, "rating": 4.5, "stock": 100},
                {"id": "p002", "name": "智能手机B", "brand": "品牌Y", "price": 3999, "rating": 4.7, "stock": 50},
                {"id": "p003", "name": "智能手机C", "brand": "品牌Z", "price": 1999, "rating": 4.2, "stock": 200}
            ],
            "笔记本电脑": [
                {"id": "l001", "name": "轻薄本A", "brand": "品牌X", "price": 5999, "rating": 4.6, "stock": 30},
                {"id": "l002", "name": "游戏本B", "brand": "品牌Y", "price": 7999, "rating": 4.8, "stock": 20},
                {"id": "l003", "name": "商务本C", "brand": "品牌Z", "price": 4999, "rating": 4.4, "stock": 50}
            ],
            "耳机": [
                {"id": "h001", "name": "无线耳机A", "brand": "品牌X", "price": 999, "rating": 4.3, "stock": 200},
                {"id": "h002", "name": "降噪耳机B", "brand": "品牌Y", "price": 1499, "rating": 4.6, "stock": 100},
                {"id": "h003", "name": "运动耳机C", "brand": "品牌Z", "price": 299, "rating": 4.1, "stock": 300}
            ],
            "平板电脑": [
                {"id": "t001", "name": "平板A", "brand": "品牌X", "price": 3499, "rating": 4.5, "stock": 50},
                {"id": "t002", "name": "平板B", "brand": "品牌Y", "price": 4499, "rating": 4.7, "stock": 30},
                {"id": "t003", "name": "平板C", "brand": "品牌Z", "price": 2499, "rating": 4.3, "stock": 100}
            ]
        }
        
        # 模拟订单数据库
        self.orders_db = {
            "o001": {"user_id": "u001", "products": [{"id": "p001", "quantity": 1}], "status": "已发货", "total": 2999},
            "o002": {"user_id": "u001", "products": [{"id": "h002", "quantity": 1}], "status": "待付款", "total": 1499},
            "o003": {"user_id": "u002", "products": [{"id": "l002", "quantity": 1}], "status": "已完成", "total": 7999}
        }

        # 添加购物车数据结构
        self.shopping_carts = {
            "u001": {
                "items": [],
                "total": 0
            }
        }
        
        # 添加大模型配置
        self.llm_config = config.get("llm", {})
        
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
        
        # 添加响应时间监控
        self.response_times = []
        
        # 添加关键词权重
        self.keywords_weight = {
            "订单": 1.0,
            "物流": 0.9,
            "发货": 0.9,
            "商品": 0.8,
            "价格": 0.8,
            "库存": 0.7,
            "购物车": 1.0
        }
    
    def process(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理电商相关的查询
        
        Args:
            user_input: 用户输入文本
            context: 上下文信息
            
        Returns:
            处理结果
        """
        start_time = time.time()
        try:
            # 添加输入验证
            if user_input is None or not isinstance(user_input, str):
                raise ValueError("输入必须是非空字符串")
            
            if context is None:
                context = {}
                
            # 把用户输入添加到context
            context["query"] = user_input
            
            result = {
                "success": True,
                "query_type": "",
                "response": "",
                "processing_time": 0
            }
            
            self.logger.info(f"处理电商查询: {user_input}")
            
            # 提取关键词
            keywords = extract_keywords(user_input)
            self.logger.debug(f"提取的关键词: {keywords}")
            
            # 分析查询类型
            query_type, category = self._analyze_query(user_input, keywords)
            result["query_type"] = query_type
            
            # 优化订单查询处理
            if query_type == "order_query":
                result["response"] = self._query_order(user_input, context)
            else:
                # 根据查询类型生成回复
                if query_type == "product_search":
                    result["response"] = self._search_products(category, keywords)
                elif query_type == "product_recommendation":
                    result["response"] = self._recommend_products(category, context)
                elif query_type == "shopping_guide":
                    result["response"] = self._provide_shopping_guide(category)
                elif query_type == "shopping_cart":
                    result["response"] = self._handle_cart_query(user_input, context)
                else:
                    result["response"] = self._general_ecommerce_response(user_input)
            
            result["processing_time"] = time.time() - start_time
            self.response_times.append(result["processing_time"])
            return result
            
        except Exception as e:
            self.logger.error(f"处理查询失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _analyze_query(self, query: str, keywords: List[str]) -> tuple:
        """
        分析查询类型和商品类别
        
        Args:
            query: 用户查询
            keywords: 提取的关键词
            
        Returns:
            (查询类型, 商品类别) 元组
        """
        # 优化订单查询识别
        if any(word in query for word in ["订单", "物流", "发货"]) or "o" in query.lower():
            return "order_query", None

        # 识别商品类别
        category = None
        for keyword in keywords:
            if keyword in self.products_db:
                category = keyword
                break
        
        # 识别查询类型
        if "推荐" in query or "有什么好" in query or "哪个好" in query:
            query_type = "product_recommendation"
        elif "搜索" in query or "查找" in query or "找" in query or "有没有" in query:
            query_type = "product_search"
        elif "怎么选" in query or "如何挑选" in query or "购买建议" in query:
            query_type = "shopping_guide"
        elif "购物车" in query:
            query_type = "shopping_cart"
        else:
            query_type = "general"
        
        return query_type, category

    def _query_order(self, query: str, context: Dict[str, Any]) -> str:
        """
        查询订单信息
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            订单查询结果
        """
        # 提取订单号
        import re
        order_id = None
        order_match = re.search(r'[o]\d{3}', query)
        if order_match:
            order_id = order_match.group()
        
        if not order_id:
            return "抱歉，没有找到订单号，请提供正确的订单号。"
            
        # 查询订单
        order = self.orders_db.get(order_id)
        if not order:
            return f"抱歉，未找到订单 {order_id} 的信息。"
            
        # 生成回复
        products_info = []
        for product in order["products"]:
            product_detail = self.get_product_details(product["id"])
            if product_detail:
                products_info.append(
                    f"{product_detail['name']} x {product['quantity']}"
                )
                
        response = f"""
订单号: {order_id}
状态: {order['status']}
商品: {', '.join(products_info)}
总金额: ¥{order['total']}

"""
        # 根据订单状态提供更详细的信息
        if order['status'] == "已发货":
            response += "您的订单已发货，请耐心等待送达。"
        elif order['status'] == "待付款":
            response += "订单尚未支付，请及时完成付款。"
        elif order['status'] == "已完成":
            response += "订单已完成，如有问题请联系客服。"
            
        return response

    def _handle_cart_query(self, query: str, context: Dict[str, Any]) -> str:
        """处理购物车相关查询"""
        user_id = context.get("user_id", "u001")
        if user_id not in self.shopping_carts:
            self.shopping_carts[user_id] = {"items": [], "total": 0}
        
        cart = self.shopping_carts[user_id]
        return self._generate_cart_response(query, cart)

    def _generate_cart_response(self, query: str, cart: Dict) -> str:
        """使用模型生成购物车相关回答"""
        try:
            # 根据context选择模型
            model_option = "自动（智能选择）"  # 默认使用自动选择
            strategy = self.model_selection_strategies[model_option]
            
            context = {
                "cart_items": cart["items"],
                "cart_total": cart["total"],
                "query": query
            }
            
            prompt = f"""
            你现在是一个专业的电商客服助手。
            用户问题：{query}
            购物车信息：{json.dumps(context, ensure_ascii=False)}
            请根据以上信息，生成专业、友好的回答，解释购物车状态并提供帮助。
            """
            
            response = ""
            for chunk in strategy(prompt, [], {}):
                response += chunk
            return response
            
        except Exception as e:
            self.logger.error(f"生成购物车回答失败: {str(e)}")
            return "抱歉，处理购物车查询时遇到问题。请稍后再试或联系客服寻求帮助。"

    def _auto_select_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Iterator[str]:
        """智能选择模型"""
        # 电商相关关键词
        commerce_keywords = ["价格", "优惠", "库存", "发货", "退款", "商品", "购物"]
        
        if any(keyword in user_input for keyword in commerce_keywords):
            return self._use_qwen_model(user_input, conversation_history, context)
        else:
            return self._use_deepseek_model(user_input, conversation_history, context)

    def _use_qwen_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Iterator[str]:
        """使用Qwen2.5模型"""
        return self.qwen_model.generate(user_input, conversation_history)

    def _use_deepseek_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Iterator[str]:
        """使用DeepSeek模型"""
        system_prompt = {
            "role": "system",
            "content": "你是一个专业的电商客服助手，擅长解答购物、物流、售后等问题。请提供准确、专业的回答。"
        }
        messages = [system_prompt]
        if conversation_history:
            messages.extend(conversation_history)
        
        return self.deepseek_model.generate(user_input, messages)

    def _use_hybrid_model(self, user_input: str, conversation_history: List[Dict[str, str]], context: Dict[str, Any]) -> Iterator[str]:
        """混合模式"""
        try:
            # 使用Qwen2.5生成基础回复
            qwen_response = ""
            for chunk in self.qwen_model.generate(user_input, conversation_history):
                qwen_response += chunk

            # 使用DeepSeek生成补充信息
            deepseek_prompt = f"请对以下电商问题提供专业的补充说明：{user_input}\n原始回答：{qwen_response}"
            deepseek_response = ""
            for chunk in self.deepseek_model.generate(deepseek_prompt, conversation_history):
                deepseek_response += chunk

            # 组合响应
            combined = f"综合回复：\n\n{qwen_response}\n\n补充信息：\n{deepseek_response}"
            for char in combined:
                yield char
                
        except Exception as e:
            self.logger.error(f"混合模式处理失败: {str(e)}")
            yield f"抱歉，处理失败: {str(e)}"

    def add_to_cart(self, user_id: str, product_id: str, quantity: int = 1) -> Dict[str, Any]:
        """添加商品到购物车"""
        if user_id not in self.shopping_carts:
            self.shopping_carts[user_id] = {"items": [], "total": 0}
        
        cart = self.shopping_carts[user_id]
        product = self.get_product_details(product_id)
        
        if product:
            # 检查库存
            if product["stock"] < quantity:
                return {"success": False, "message": "库存不足"}
            
            # 检查是否已在购物车中
            for item in cart["items"]:
                if item["id"] == product_id:
                    item["quantity"] += quantity
                    cart["total"] += product["price"] * quantity
                    return {"success": True, "cart": cart}
            
            # 添加新商品
            cart["items"].append({
                "id": product_id,
                "name": product["name"],
                "price": product["price"],
                "quantity": quantity
            })
            cart["total"] += product["price"] * quantity
            return {"success": True, "cart": cart}
        
        return {"success": False, "message": "商品不存在"}

    def _general_ecommerce_response(self, query: str) -> str:
        """使用大模型生成通用电商回答"""
        try:
            # 使用正确的导入路径
            model = self.model_selection_strategies["自动（智能选择）"](query, [], {})
            
            prompt = f"""
            作为电商助手，请回答用户的问题：{query}
            回答要求：
            1. 专业且友好
            2. 提供具体的帮助和建议
            3. 介绍可用的功能（如搜索商品、查询订单、购物车管理等）
            """
            
            response = ""
            for chunk in model:
                response += chunk
            return response
            
        except Exception as e:
            self.logger.error(f"生成回答失败: {str(e)}")
            return "抱歉，我暂时无法回答您的问题。请稍后再试或联系客服寻求帮助。"

    def _search_products(self, category: str, keywords: List[str]) -> str:
        """搜索商品并进行性价比分析"""
        try:
            if not category:
                return "请告诉我您想搜索哪类商品？我们有手机、笔记本电脑、耳机和平板电脑等类别。"
            
            # 性价比分析维度和权重
            value_metrics = {
                "手机": {
                    "price": {"weight": 0.3, "description": "价格区间"},
                    "rating": {"weight": 0.2, "description": "用户评分"},
                    "brand": {"weight": 0.2, "description": "品牌口碑"},
                    "stock": {"weight": 0.1, "description": "库存状况"},
                    "price_per_rating": {"weight": 0.2, "description": "性价比指数"}
                }
            }
            
            products = self.products_db.get(category, [])
            if not products:
                return f"抱歉，未找到{category}类别的商品。"
              
            # 如果是手机类别，进行性价比分析
            if category == "手机":
                # 计算每个产品的性价比指数
                for product in products:
                    # 价格得分（价格越低分数越高）
                    max_price = max(p["price"] for p in products)
                    min_price = min(p["price"] for p in products)
                    price_score = 1 - (product["price"] - min_price) / (max_price - min_price) if max_price != min_price else 1
                    
                    # 评分得分（直接使用评分）
                    rating_score = product["rating"] / 5.0
                    
                    # 品牌得分（简单示例，实际应该基于品牌数据）
                    brand_score = 0.8  # 假设所有品牌都有基础分
                    
                    # 库存得分（库存充足度）
                    stock_score = min(product["stock"] / 100, 1.0)  # 假设100是理想库存
                    
                    # 性价比指数计算
                    value_metrics = {
                        "price": {"score": price_score, "weight": 0.3},
                        "rating": {"score": rating_score, "weight": 0.2},
                        "brand": {"score": brand_score, "weight": 0.2},
                        "stock": {"score": stock_score, "weight": 0.1},
                        "price_per_rating": {"score": rating_score / (product["price"] / 1000), "weight": 0.2}
                    }
                    
                    # 计算总分
                    product["value_score"] = sum(metric["score"] * metric["weight"] for metric in value_metrics.values())
                
                # 按性价比排序
                products.sort(key=lambda x: x["value_score"], reverse=True)
                
                # 生成分析报告
                report = f"为您找到{len(products)}款{category}，按性价比从高到低排序：\n\n"
                for i, product in enumerate(products, 1):
                    report += f"{i}. {product['name']} ({product['brand']})\n"
                    report += f"   价格：¥{product['price']}\n"
                    report += f"   评分：{product['rating']}星\n"
                    report += f"   库存：{product['stock']}件\n"
                    report += f"   性价比指数：{product['value_score']:.2f}\n"
                    report += "   -------------\n"
                
                return report

            # 解析价格范围
            price_range = self._parse_price_range(" ".join(keywords))
            if price_range:
                products = [p for p in products if price_range[0] <= p["price"] <= price_range[1]]

            if not products:
                return f"抱歉，没有找到符合条件的{category}。"

            response = f"为您找到以下{category}：\n\n"
            for product in products:
                response += f"- {product['name']}\n"
                response += f"  品牌：{product['brand']}\n"
                response += f"  价格：¥{product['price']}\n"
                response += f"  评分：{product['rating']}\n"
                response += f"  库存：{product['stock']}\n\n"
            return response

        except Exception as e:
            self.logger.error(f"搜索商品失败: {str(e)}")
            return "抱歉，搜索商品时遇到问题。请稍后再试。"

    def _recommend_products(self, category: str, context: Dict[str, Any]) -> str:
        """推荐商品"""
        try:
            query = context.get("query", "").lower()
            
            # 如果query中包含商品类别，优先使用
            for cat in self.products_db.keys():
                if cat in query:
                    category = cat
                    break
            
            if not category:
                return "请告诉我您对哪类商品感兴趣？我们可以推荐手机、笔记本电脑、耳机和平板电脑等。"

            products = self.products_db.get(category, [])
            if not products:
                return f"抱歉，暂时没有{category}的推荐。"

            # 获取价格范围（如果有）
            price_range = self._parse_price_range(query)
            if price_range:
                products = [p for p in products if price_range[0] <= p["price"] <= price_range[1]]
                
                # 先按价格筛选，再按评分排序
                sorted_products = sorted(products, key=lambda x: x["rating"], reverse=True)
            else:
                # 没有价格要求时，按评分和价格综合排序
                sorted_products = sorted(products, key=lambda x: (x["rating"], -x["price"]), reverse=True)

            if not sorted_products:
                return f"抱歉，没有找到符合价格要求的{category}推荐。"

            # 检查是否是促销查询
            is_promotion_query = "促销" in query or "优惠" in query or "活动" in query
            
            # 推荐前3个商品
            response = f"根据您的需求，为您推荐以下{category}"
            response += "促销商品" if is_promotion_query else "商品"
            response += "：\n\n"
            
            for product in sorted_products[:3]:
                response += f"▶ {product['name']}\n"
                response += f"  - 品牌：{product['brand']}\n"
                response += f"  - 价格：¥{product['price']}"
                
                # 添加促销信息
                if is_promotion_query:
                    if product['price'] >= 1000:
                        response += f" (限时优惠：立减¥{int(product['price']*0.1)})"
                    else:
                        response += " (限时9折优惠)"
                response += "\n"
                
                response += f"  - 评分：{product['rating']}分\n"
                response += f"  - 库存：{product['stock']}件\n"
                response += f"  - 特点：{self._get_product_features(product)}\n\n"
                
            # 添加附加建议
            if price_range:
                response += f"\n以上是{price_range[1]}元以下的{category}推荐，如果预算可以提高，还有更多优选商品供您参考。"
            
            # 添加促销活动说明
            if is_promotion_query:
                response += "\n\n当前促销活动：\n"
                response += "1. 千元以上商品立减10%\n"
                response += "2. 千元以下商品9折优惠\n"
                response += "3. 活动时间：限时特惠，欢迎咨询具体详情"
            
            return response

        except Exception as e:
            self.logger.error(f"推荐商品失败: {str(e)}")
            return "抱歉，生成推荐时遇到问题。请稍后再试。"

    def _parse_price_range(self, query: str) -> Optional[tuple]:
        """解析价格范围"""
        import re
        
        # 匹配价格范围
        price_pattern = r'(\d+)元以?下'
        match = re.search(price_pattern, query)
        if match:
            price = int(match.group(1))
            return (0, price)
            
        range_pattern = r'(\d+)[-~到至](\d+)元'
        match = re.search(range_pattern, query)
        if match:
            return (int(match.group(1)), int(match.group(2)))
            
        return None

    def _get_product_features(self, product: Dict[str, Any]) -> str:
        """获取商品特点"""
        features = []
        
        # 评分分析
        if product["rating"] >= 4.7:
            features.append("好评如潮")
        elif product["rating"] >= 4.5:
            features.append("好评优选")
        elif product["rating"] >= 4.3:
            features.append("用户认可")
        
        # 价格定位分析
        category_products = self.products_db.get(self._get_product_category(product["id"]), [])
        if category_products:
            prices = [p["price"] for p in category_products]
            avg_price = sum(prices) / len(prices)
            if product["price"] >= avg_price * 1.5:
                features.append("高端定位")
            elif product["price"] <= avg_price * 0.7:
                features.append("性价比高")
            elif avg_price * 0.7 < product["price"] < avg_price * 1.2:
                features.append("主流价位")
        
        # 库存状态分析
        if product["stock"] > 200:
            features.append("库存充足")
        elif 50 < product["stock"] <= 200:
            features.append("现货在售")
        elif 20 < product["stock"] <= 50:
            features.append("库存紧张")
        else:
            features.append("即将售罄")
        
        # 品牌特点
        if product["brand"] == "品牌X":
            features.append("科技领先")
        elif product["brand"] == "品牌Y":
            features.append("品质保证")
        elif product["brand"] == "品牌Z":
            features.append("高性价比")
            
        # 添加商品类别特点
        category = self._get_product_category(product["id"])
        if category == "手机":
            features.append("智能设备")
        elif category == "笔记本电脑":
            features.append("办公娱乐")
        elif category == "耳机":
            features.append("音频设备")
        elif category == "平板电脑":
            features.append("便携办公")
            
        return "、".join(features) if features else "暂无特点描述"

    def _get_product_category(self, product_id: str) -> Optional[str]:
        """根据商品ID获取商品类别"""
        for category, products in self.products_db.items():
            if any(p["id"] == product_id for p in products):
                return category
        return None

    def _provide_shopping_guide(self, category: str) -> Iterator[str]:
        """提供购物指南"""
        guides = {
            "手机": """手机选购指南

主要考虑因素：
1. 性能配置：处理器、内存、存储空间
2. 拍照功能：摄像头参数、防抖、夜拍
3. 电池续航：电池容量、快充技术
4. 屏幕品质：分辨率、刷新率、显示技术
5. 手机尺寸：重量、握持感、便携性

选购建议：
• 明确预算和使用需求
• 对比不同品牌型号
• 查看用户真实评价
• 关注售后服务政策""",
            "笔记本电脑": """笔记本电脑选购指南

主要考虑因素：
1. 处理器性能：CPU型号、核心数
2. 显卡配置：独立显卡/集成显卡
3. 内存容量：建议8GB起步
4. 存储方案：SSD+HDD组合
5. 屏幕素质：分辨率、色域、亮度

选购建议：
• 根据使用场景选择
• 注意散热设计
• 考虑接口扩展性
• 选择合适的重量
• 关注续航能力
• 确认保修政策""",
            "耳机": """耳机选购指南

主要考虑因素：
1. 佩戴方式：入耳式/头戴式
2. 连接方式：有线/无线
3. 音质表现：频响范围、降噪
4. 续航时间：电池容量、充电速度
5. 防水防汗：运动使用需求

选购建议：
• 确定使用场景
• 试听音质效果
• 检查佩戴舒适度
• 了解售后保障""",
            "平板电脑": """平板电脑选购指南

主要考虑因素：
1. 屏幕大小：便携性与显示效果
2. 系统生态：应用商店资源
3. 处理性能：办公/娱乐需求
4. 配件支持：手写笔/键盘
5. 续航能力：电池容量

选购建议：
• 明确使用目的
• 考虑扩展性能
• 对比不同品牌
• 关注系统更新
• 评估配件成本"""
        }
        
        if category not in guides:
            yield "抱歉，暂时没有该类商品的购物指南。我们目前提供手机、笔记本电脑、耳机和平板电脑的选购建议。"
            return
            
        for char in guides[category]:
            yield char