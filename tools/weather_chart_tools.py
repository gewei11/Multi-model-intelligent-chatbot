from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from PIL import Image, ImageDraw, ImageFont
import os

@dataclass
class WeatherChartTool:
    """天气图表绘制工具"""
    name: str = "weather_chart"
    description: str = "绘制天气数据的折线图，包括温度曲线等"
    
    def __init__(self):
        """初始化图表绘制工具"""
        self.canvas_size = (800, 600)  # 默认画布大小
        self.margin = 60  # 图表边距
        self.canvas = Image.new('RGB', self.canvas_size, 'white')
        self.draw = ImageDraw.Draw(self.canvas)
        
        # 尝试加载字体，如果失败则使用默认字体
        try:
            self.font = ImageFont.truetype("simhei.ttf", 12)  # 中文字体
        except:
            self.font = ImageFont.load_default()
    
    def draw_temperature_chart(self, dates: List[str], high_temps: List[int], low_temps: List[int]) -> str:
        """绘制温度折线图
        
        Args:
            dates: 日期列表
            high_temps: 最高温度列表
            low_temps: 最低温度列表
            
        Returns:
            str: 绘图结果描述
        """
        # 重置画布
        self.canvas = Image.new('RGB', self.canvas_size, 'white')
        self.draw = ImageDraw.Draw(self.canvas)
        
        # 计算数据范围
        max_temp = max(max(high_temps), max(low_temps)) + 5
        min_temp = min(min(high_temps), min(low_temps)) - 5
        
        # 计算绘图区域
        chart_width = self.canvas_size[0] - 2 * self.margin
        chart_height = self.canvas_size[1] - 2 * self.margin
        
        # 绘制坐标轴
        self.draw.line([(self.margin, self.margin), 
                       (self.margin, self.canvas_size[1] - self.margin)], 
                      fill='black', width=2)  # Y轴
        self.draw.line([(self.margin, self.canvas_size[1] - self.margin), 
                       (self.canvas_size[0] - self.margin, self.canvas_size[1] - self.margin)], 
                      fill='black', width=2)  # X轴
        
        # 绘制温度刻度和网格线
        temp_range = max_temp - min_temp
        for i in range(int(min_temp), int(max_temp) + 1, 5):
            y = self.canvas_size[1] - self.margin - (i - min_temp) / temp_range * chart_height
            # 绘制刻度
            self.draw.text((self.margin - 25, y - 6), f"{i}°C", fill='black', font=self.font)
            self.draw.line([(self.margin - 5, y), (self.margin, y)], fill='black', width=1)
            # 绘制网格线
            self.draw.line([(self.margin, y), (self.canvas_size[0] - self.margin, y)], 
                          fill='lightgray', width=1)
        
        # 绘制日期标签
        x_step = chart_width / (len(dates) - 1)
        for i, date in enumerate(dates):
            x = self.margin + i * x_step
            # 提取日期中的月日
            display_date = date.split('-')[1] + '/' + date.split('-')[2]
            self.draw.text((x - 15, self.canvas_size[1] - self.margin + 10), 
                          display_date, fill='black', font=self.font)
            # 绘制垂直网格线
            self.draw.line([(x, self.margin), (x, self.canvas_size[1] - self.margin)], 
                          fill='lightgray', width=1)
        
        # 绘制温度曲线
        high_points = []
        low_points = []
        for i in range(len(dates)):
            x = self.margin + i * x_step
            # 最高温度点
            y_high = self.canvas_size[1] - self.margin - \
                    (high_temps[i] - min_temp) / temp_range * chart_height
            high_points.append((x, y_high))
            # 最低温度点
            y_low = self.canvas_size[1] - self.margin - \
                   (low_temps[i] - min_temp) / temp_range * chart_height
            low_points.append((x, y_low))
        
        # 绘制曲线和数据点
        self.draw.line(high_points, fill='red', width=2)
        self.draw.line(low_points, fill='blue', width=2)
        
        # 绘制温度数据点和标签
        for i in range(len(dates)):
            # 最高温度点
            self.draw.ellipse([(high_points[i][0] - 4, high_points[i][1] - 4),
                              (high_points[i][0] + 4, high_points[i][1] + 4)], 
                             fill='red')
            self.draw.text((high_points[i][0] - 10, high_points[i][1] - 20),
                          f"{high_temps[i]}°", fill='red', font=self.font)
            
            # 最低温度点
            self.draw.ellipse([(low_points[i][0] - 4, low_points[i][1] - 4),
                              (low_points[i][0] + 4, low_points[i][1] + 4)], 
                             fill='blue')
            self.draw.text((low_points[i][0] - 10, low_points[i][1] + 10),
                          f"{low_temps[i]}°", fill='blue', font=self.font)
        
        # 添加图例
        legend_x = self.canvas_size[0] - self.margin - 100
        legend_y = self.margin + 20
        self.draw.line([(legend_x, legend_y), (legend_x + 20, legend_y)], fill='red', width=2)
        self.draw.text((legend_x + 30, legend_y - 6), "最高温", fill='red', font=self.font)
        self.draw.line([(legend_x, legend_y + 20), (legend_x + 20, legend_y + 20)], 
                      fill='blue', width=2)
        self.draw.text((legend_x + 30, legend_y + 14), "最低温", fill='blue', font=self.font)
        
        # 添加标题
        title = "未来天气温度变化趋势"
        self.draw.text((self.canvas_size[0] // 2 - 80, self.margin - 30), 
                      title, fill='black', font=self.font)
        
        return "成功绘制天气温度折线图"
    
    def save_image(self, file_path: str):
        """保存图表为图片"""
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self.canvas.save(file_path)