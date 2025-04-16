import sys
import os
import unittest
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.domain_agents.ecommerce_agent import EcommerceAgent

class TestEcommerceAgent(unittest.TestCase):
    def setUp(self):
        """测试前准备"""
        self.config = {
            "models": {
                "qwen": {
                    "model_name": "qwen2.5:7b",
                    "api_key": "test_key"
                },
                "deepseek": {
                    "model_name": "MFDoom/deepseek-r1-tool-calling:8b",
                    "api_key": "test_key"
                }
            }
        }
        self.agent = EcommerceAgent(self.config)
        
    def print_test_result(self, test_name: str, response: str):
        """打印测试结果"""
        print(f"\n=== {test_name} ===")
        print(f"回答: {response}\n")

    def test_provide_shopping_guide(self):
        """测试购物指南功能"""
        # 测试手机购物指南
        response = "".join(self.agent._provide_shopping_guide("手机"))
        self.print_test_result("购物指南 - 手机", response)
        self.assertIn("选购指南", response)
        self.assertIn("性能配置", response)
        self.assertIn("主要考虑因素", response)
        
        # 测试不存在的类别
        response = "".join(self.agent._provide_shopping_guide("不存在的类别"))
        self.print_test_result("购物指南 - 不存在类别", response)
        self.assertIn("请告诉我您想了解哪类商品的购买建议", response)

    def test_recommend_products(self):
        """测试商品推荐功能"""
        context = {"query": "想买一个2000元以下的手机"}
        response = self.agent._recommend_products("手机", context)
        self.print_test_result("商品推荐", response)
        
        self.assertIn("根据您的需求，推荐以下手机", response)
        self.assertIn("品牌", response)
        self.assertIn("价格", response)
        self.assertIn("评分", response)

    def test_search_products(self):
        """测试商品搜索功能"""
        keywords = ["手机", "2000"]
        response = self.agent._search_products("手机", keywords)
        self.print_test_result("商品搜索", response)
        
        self.assertIn("为您找到以下手机", response)
        self.assertIn("品牌", response)
        self.assertIn("价格", response)
        self.assertIn("库存", response)

    def test_parse_price_range(self):
        """测试价格范围解析"""
        # 测试"xx元以下"格式
        result = self.agent._parse_price_range("2000元以下")
        self.print_test_result("价格范围解析 - xx元以下", str(result))
        self.assertEqual(result, (0, 2000))
        
        # 测试"xx到xx元"格式
        result = self.agent._parse_price_range("1000到3000元")
        self.print_test_result("价格范围解析 - xx到xx元", str(result))
        self.assertEqual(result, (1000, 3000))
        
        # 测试无价格信息
        result = self.agent._parse_price_range("手机推荐")
        self.print_test_result("价格范围解析 - 无价格信息", str(result))
        self.assertIsNone(result)

    def test_query_order(self):
        """测试订单查询功能"""
        # 测试有效订单号
        response = self.agent._query_order("查询订单o001", {})
        self.print_test_result("订单查询 - 有效订单", response)
        self.assertIn("订单号: o001", response)
        self.assertIn("状态", response)
        self.assertIn("总金额", response)
        
        # 测试无效订单号
        response = self.agent._query_order("查询订单xxx", {})
        self.print_test_result("订单查询 - 无效订单", response)
        self.assertIn("抱歉", response)

    def test_process_model_output(self):
        """测试模型输出处理"""
        # 测试字符串输入
        output = list(self.agent._process_model_output("test message"))
        self.print_test_result("模型输出处理 - 字符串输入", "".join(output))
        self.assertEqual("".join(output), "test message")
        
        # 测试带message的字典输入
        test_dict = {"message": {"content": "test content"}}
        # 使用迭代器处理输出
        output = "".join(list(self.agent._process_model_output([test_dict])))
        self.print_test_result("模型输出处理 - 字典输入", output)
        self.assertEqual(output, "test content")

    def test_get_product_features(self):
        """测试商品特点提取"""
        product = {
            "rating": 4.7,
            "stock": 150,
            "price": 1999
        }
        features = self.agent._get_product_features(product)
        self.print_test_result("商品特点提取", features)
        
        self.assertIn("好评率高", features)
        self.assertIn("库存充足", features)
        self.assertIn("价格实惠", features)

    def test_analyze_query(self):
        """测试查询分析"""
        query = "推荐2000元以下的手机"
        keywords = ["手机", "2000", "推荐"]
        query_type, category = self.agent._analyze_query(query, keywords)
        self.print_test_result("查询分析", f"查询类型: {query_type}, 商品类别: {category}")
        
        self.assertEqual(query_type, "product_recommendation")
        self.assertEqual(category, "手机")

    def test_error_handling(self):
        """测试错误处理"""
        # 触发处理失败异常
        try:
            # 传入None作为参数来触发异常
            result = self.agent.process(None)
            self.print_test_result("错误处理", f"处理结果: {result}")
            # 检查是否包含错误信息
            self.assertFalse(result["success"])
            self.assertIn("error", result)
            self.assertIn("processing_time", result)
        except Exception as e:
            self.fail(f"测试过程中出现意外错误: {str(e)}")

if __name__ == "__main__":
    unittest.main(verbosity=2)
