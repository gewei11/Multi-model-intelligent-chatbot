import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.drawing_tools import LineChartTool
import matplotlib.pyplot as plt

def test_basic_line_chart():
    """测试基本折线图功能"""
    line_chart = LineChartTool()
    
    # 创建单条折线的数据
    data_series = [
        {
            "name": "销售额",
            "color": "red",
            "x_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "y_values": [120, 150, 130, 190, 210, 170, 230, 240, 200, 180, 220, 250]
        }
    ]
    
    # 绘制折线图
    result = line_chart.draw_chart(
        title="月度销售额分析",
        data_series=data_series,
        x_label="月份",
        y_label="销售额（万元）"
    )
    
    print(result)
    
    # 保存图表
    output_path = os.path.join("output", "sales_chart.png")
    line_chart.save_image(output_path)
    print(f"图表已保存至：{output_path}")

def test_multi_line_chart():
    """测试多条折线图功能"""
    line_chart = LineChartTool()
    
    # 创建多条折线的数据
    data_series = [
        {
            "name": "销售额",
            "color": "red",
            "x_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "y_values": [120, 150, 130, 190, 210, 170, 230, 240, 200, 180, 220, 250]
        },
        {
            "name": "成本",
            "color": "blue",
            "x_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "y_values": [80, 90, 85, 100, 110, 95, 120, 125, 105, 95, 115, 130]
        },
        {
            "name": "利润",
            "color": "green",
            "x_values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            "y_values": [40, 60, 45, 90, 100, 75, 110, 115, 95, 85, 105, 120]
        }
    ]
    
    # 绘制折线图
    result = line_chart.draw_chart(
        title="销售、成本与利润分析",
        data_series=data_series,
        x_label="月份",
        y_label="金额（万元）"
    )
    
    print(result)
    
    # 保存图表
    output_path = os.path.join("output", "sales_cost_profit_chart.png")
    line_chart.save_image(output_path)
    print(f"图表已保存至：{output_path}")

def test_custom_chart():
    """测试自定义参数的折线图"""
    line_chart = LineChartTool()
    
    # 创建自定义数据
    data_series = [
        {
            "name": "产品A",
            "color": "purple",
            "x_values": [2020, 2021, 2022, 2023, 2024],
            "y_values": [50, 65, 80, 95, 110]
        },
        {
            "name": "产品B",
            "color": "orange",
            "x_values": [2020, 2021, 2022, 2023, 2024],
            "y_values": [30, 45, 70, 85, 100]
        }
    ]
    
    # 绘制折线图
    result = line_chart.draw_chart(
        title="产品销量趋势分析",
        data_series=data_series,
        x_label="年份",
        y_label="销量（万件）",
        canvas_size=[1000, 700]  # 自定义画布大小
    )
    
    print(result)
    
    # 保存图表
    output_path = os.path.join("output", "product_trend_chart.png")
    line_chart.save_image(output_path)
    print(f"图表已保存至：{output_path}")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs("output", exist_ok=True)
    
    print("测试基本折线图...")
    test_basic_line_chart()
    
    print("\n测试多条折线图...")
    test_multi_line_chart()
    
    print("\n测试自定义参数的折线图...")
    test_custom_chart()
    
    print("\n所有测试完成！")