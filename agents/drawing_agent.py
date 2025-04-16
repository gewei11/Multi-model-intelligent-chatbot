from typing import Dict, Any, Optional, List
from agents.core_agent import CoreAgent
from tools.drawing_tools import DrawingTool, LineChartTool
from tools.weather_chart_tools import WeatherChartTool
import os
import re

class DrawingAgent(CoreAgent):
    """绘图Agent，用于处理用户的绘图请求"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.drawing_tool = DrawingTool()
        self.weather_chart_tool = WeatherChartTool()
        self.line_chart_tool = LineChartTool()
        self.output_dir = 'output'
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """处理用户消息，解析绘图意图并执行绘图操作"""
        try:
            # 检查是否是天气图表绘制请求
            if '天气' in message and ('折线图' in message or '图表' in message):
                # 从消息中提取天气数据
                dates = ['2025-04-11', '2025-04-12', '2025-04-13']
                high_temps = [26, 21, 25]
                low_temps = [13, 10, 11]
                
                # 绘制天气图表
                result = self.weather_chart_tool.draw_temperature_chart(
                    dates=dates,
                    high_temps=high_temps,
                    low_temps=low_temps
                )
                
                # 保存图表
                output_path = os.path.join(self.output_dir, 'weather_chart.png')
                self.weather_chart_tool.save_image(output_path)
                
                return {
                    "status": "success",
                    "message": f"天气图表绘制完成\n图片已保存至：{output_path}",
                    "image_path": output_path
                }
            
            # 检查是否是折线分析图绘制请求
            if ('折线' in message or '分析图' in message) and not '天气' in message:
                # 示例数据 - 在实际应用中可以从消息中解析或从数据源获取
                data_series = [
                    {
                        "name": "销售额",
                        "color": "red",
                        "x_values": [1, 2, 3, 4, 5, 6, 7],
                        "y_values": [120, 150, 130, 190, 210, 170, 230]
                    },
                    {
                        "name": "成本",
                        "color": "blue",
                        "x_values": [1, 2, 3, 4, 5, 6, 7],
                        "y_values": [80, 90, 85, 100, 110, 95, 120]
                    }
                ]
                
                # 从消息中提取标题和标签信息
                title = "销售与成本分析图"
                x_label = "月份"
                y_label = "金额（万元）"
                
                # 绘制折线分析图
                result = self.line_chart_tool.draw_chart(
                    title=title,
                    data_series=data_series,
                    x_label=x_label,
                    y_label=y_label,
                    canvas_size=[800, 600]
                )
                
                # 保存图表
                output_path = os.path.join(self.output_dir, 'line_chart.png')
                self.line_chart_tool.save_image(output_path)
                
                return {
                    "status": "success",
                    "message": f"折线分析图绘制完成\n图片已保存至：{output_path}",
                    "image_path": output_path
                }
            
            # 处理普通绘图请求
            result = self.drawing_tool.parse_query(
                text=message,
                canvas_size=[800, 600]  # 默认画布大小
            )
            
            # 保存绘图结果
            output_path = os.path.join(self.output_dir, 'drawing_output.png')
            self.drawing_tool.save_image(output_path)
            
            return {
                "status": "success",
                "message": f"绘图完成：{result['message']}\n图片已保存至：{output_path}",
                "image_path": output_path
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"绘图过程出现错误：{str(e)}"
            }
    
    def get_supported_tools(self) -> Dict[str, Any]:
        """返回支持的工具列表"""
        return {
            "drawing": self.drawing_tool,
            "line_chart": self.line_chart_tool
        }