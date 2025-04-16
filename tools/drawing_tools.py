from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import re
import math
from PIL import Image, ImageDraw, ImageFont
import os

@dataclass
class DrawingTool:
    """绘图工具"""
    name: str = "drawing"
    description: str = "支持基本图形绘制，包括线条、矩形、圆形等，可以设置颜色和线条粗细"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "绘图指令，如：画一条红色线条"
            },
            "canvas_size": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "画布大小，格式为[宽度, 高度]",
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["command", "canvas_size"]
    })
    
    def __post_init__(self):
        """初始化画布和绘图状态"""
        self.canvas_size = (800, 600)  # 默认画布大小
        self.canvas = Image.new('RGB', self.canvas_size, 'white')
        self.draw = ImageDraw.Draw(self.canvas)
        self.current_color = 'black'  # 默认颜色
        self.current_width = 2  # 默认线条宽度
    
    def parse_color(self, text: str) -> str:
        """解析颜色信息"""
        color_map = {
            '红': 'red', '红色': 'red',
            '蓝': 'blue', '蓝色': 'blue',
            '绿': 'green', '绿色': 'green',
            '黄': 'yellow', '黄色': 'yellow',
            '黑': 'black', '黑色': 'black',
            '白': 'white', '白色': 'white'
        }
        for cn_color, en_color in color_map.items():
            if cn_color in text:
                return en_color
        return self.current_color
    
    def parse_width(self, text: str) -> int:
        """解析线条宽度"""
        match = re.search(r'(\d+)像素', text)
        if match:
            return int(match.group(1))
        if '粗' in text:
            return 4
        if '细' in text:
            return 1
        return self.current_width
    
    def parse_coordinates(self, text: str) -> List[Tuple[int, int]]:
        """解析坐标信息"""
        coordinates = []
        matches = re.finditer(r'\((\d+)[,，](\d+)\)', text)
        for match in matches:
            x, y = int(match.group(1)), int(match.group(2))
            coordinates.append((x, y))
        return coordinates
    
    def draw_line(self, start: Tuple[int, int], end: Tuple[int, int]):
        """画线"""
        self.draw.line([start, end], fill=self.current_color, width=self.current_width)
    
    def draw_rectangle(self, start: Tuple[int, int], end: Tuple[int, int]):
        """画矩形"""
        self.draw.rectangle([start, end], outline=self.current_color, width=self.current_width)
    
    def draw_circle(self, center: Tuple[int, int], radius: int):
        """画圆"""
        left_up = (center[0] - radius, center[1] - radius)
        right_down = (center[0] + radius, center[1] + radius)
        self.draw.ellipse([left_up, right_down], outline=self.current_color, width=self.current_width)
    
    def parse_query(self, text: str, canvas_size: List[int] = None) -> Dict[str, Any]:
        """解析绘图指令"""
        if canvas_size:
            self.canvas_size = tuple(canvas_size)
            self.canvas = Image.new('RGB', self.canvas_size, 'white')
            self.draw = ImageDraw.Draw(self.canvas)
        
        # 解析颜色和线条宽度
        self.current_color = self.parse_color(text)
        self.current_width = self.parse_width(text)
        
        # 解析坐标
        coordinates = self.parse_coordinates(text)
        
        # 根据指令类型执行相应的绘图操作
        if '线' in text and len(coordinates) >= 2:
            self.draw_line(coordinates[0], coordinates[1])
        elif '矩形' in text and len(coordinates) >= 2:
            self.draw_rectangle(coordinates[0], coordinates[1])
        elif '圆' in text and len(coordinates) >= 1:
            radius = 50  # 默认半径
            if len(coordinates) > 1:
                # 如果提供了两个点，用两点间距离作为直径
                dx = coordinates[1][0] - coordinates[0][0]
                dy = coordinates[1][1] - coordinates[0][1]
                radius = int((dx * dx + dy * dy) ** 0.5 / 2)
            self.draw_circle(coordinates[0], radius)
        
        return {
            "status": "success",
            "message": f"已完成绘制，使用{self.current_color}色，线条宽度{self.current_width}"
        }
    
    def save_image(self, file_path: str):
        """保存绘图结果"""
        self.canvas.save(file_path)


@dataclass
class LineChartTool:
    """折线图分析工具"""
    name: str = "line_chart"
    description: str = "绘制数据折线分析图，支持多条折线、坐标轴标签和图例"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "图表标题"
            },
            "x_label": {
                "type": "string",
                "description": "X轴标签"
            },
            "y_label": {
                "type": "string",
                "description": "Y轴标签"
            },
            "data_series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "数据系列名称"
                        },
                        "color": {
                            "type": "string",
                            "description": "折线颜色"
                        },
                        "x_values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "X轴数据点"
                        },
                        "y_values": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "Y轴数据点"
                        }
                    },
                    "required": ["name", "x_values", "y_values"]
                },
                "description": "数据系列列表"
            },
            "canvas_size": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "画布大小，格式为[宽度, 高度]",
                "minItems": 2,
                "maxItems": 2
            }
        },
        "required": ["title", "data_series"]
    })
    
    def __init__(self):
        """初始化折线图工具"""
        self.canvas_size = (800, 600)  # 默认画布大小
        self.margin = 60  # 图表边距
        self.canvas = Image.new('RGB', self.canvas_size, 'white')
        self.draw = ImageDraw.Draw(self.canvas)
        self.colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
        
    def draw_chart(self, title: str, data_series: List[Dict[str, Any]], 
                  x_label: str = "", y_label: str = "", 
                  canvas_size: List[int] = None) -> Dict[str, Any]:
        """绘制折线图
        
        Args:
            title: 图表标题
            data_series: 数据系列列表，每个系列包含name, color(可选), x_values, y_values
            x_label: X轴标签
            y_label: Y轴标签
            canvas_size: 画布大小
            
        Returns:
            Dict[str, Any]: 绘图结果描述
        """
        # 重置画布
        if canvas_size:
            self.canvas_size = tuple(canvas_size)
        self.canvas = Image.new('RGB', self.canvas_size, 'white')
        self.draw = ImageDraw.Draw(self.canvas)
        
        # 计算数据范围
        all_x_values = [x for series in data_series for x in series['x_values']]
        all_y_values = [y for series in data_series for y in series['y_values']]
        
        if not all_x_values or not all_y_values:
            return {"status": "error", "message": "没有提供有效的数据点"}
        
        x_min, x_max = min(all_x_values), max(all_x_values)
        y_min, y_max = min(all_y_values), max(all_y_values)
        
        # 为了美观，扩展Y轴范围
        y_range = y_max - y_min
        if y_range == 0:  # 防止除零错误
            y_range = 1
        y_min = y_min - y_range * 0.1
        y_max = y_max + y_range * 0.1
        
        # 计算绘图区域
        chart_width = self.canvas_size[0] - 2 * self.margin
        chart_height = self.canvas_size[1] - 2 * self.margin
        
        # 绘制坐标轴
        self._draw_axes(x_min, x_max, y_min, y_max, x_label, y_label)
        
        # 绘制数据系列
        for i, series in enumerate(data_series):
            color = series.get('color', self.colors[i % len(self.colors)])
            self._draw_data_series(series['x_values'], series['y_values'], 
                                  x_min, x_max, y_min, y_max,
                                  chart_width, chart_height, color)
        
        # 添加图例
        self._draw_legend(data_series)
        
        # 添加标题
        self._draw_title(title)
        
        return {"status": "success", "message": f"成功绘制折线图：{title}"}
    
    def _draw_axes(self, x_min: float, x_max: float, y_min: float, y_max: float, 
                  x_label: str, y_label: str):
        """绘制坐标轴"""
        # 绘制X轴和Y轴
        self.draw.line([(self.margin, self.margin), 
                       (self.margin, self.canvas_size[1] - self.margin)], 
                      fill='black', width=2)  # Y轴
        self.draw.line([(self.margin, self.canvas_size[1] - self.margin), 
                       (self.canvas_size[0] - self.margin, self.canvas_size[1] - self.margin)], 
                      fill='black', width=2)  # X轴
        
        # 绘制Y轴刻度
        chart_height = self.canvas_size[1] - 2 * self.margin
        y_range = y_max - y_min
        
        # 计算合适的Y轴刻度间隔
        y_step = self._calculate_step(y_min, y_max)
        
        for y in self._generate_ticks(y_min, y_max, y_step):
            y_pos = self.canvas_size[1] - self.margin - (y - y_min) / y_range * chart_height
            self.draw.line([(self.margin - 5, y_pos), (self.margin, y_pos)], fill='black', width=1)
            self.draw.text((self.margin - 40, y_pos - 10), f"{y:.1f}", fill='black')
        
        # 绘制X轴刻度
        chart_width = self.canvas_size[0] - 2 * self.margin
        x_range = x_max - x_min
        
        # 计算合适的X轴刻度间隔
        x_step = self._calculate_step(x_min, x_max)
        
        for x in self._generate_ticks(x_min, x_max, x_step):
            x_pos = self.margin + (x - x_min) / x_range * chart_width
            self.draw.line([(x_pos, self.canvas_size[1] - self.margin), 
                           (x_pos, self.canvas_size[1] - self.margin + 5)], fill='black', width=1)
            self.draw.text((x_pos - 15, self.canvas_size[1] - self.margin + 10), 
                          f"{x:.1f}", fill='black')
        
        # 添加坐标轴标签
        if x_label:
            self.draw.text((self.canvas_size[0] // 2 - 20, self.canvas_size[1] - 20), 
                          x_label, fill='black')
        if y_label:
            # 垂直绘制Y轴标签
            for i, char in enumerate(y_label):
                self.draw.text((10, self.canvas_size[1] // 2 - 10 * len(y_label) // 2 + i * 20), 
                              char, fill='black')
    
    def _calculate_step(self, min_val: float, max_val: float) -> float:
        """计算合适的刻度间隔"""
        range_val = max_val - min_val
        if range_val == 0:
            return 1.0
        
        magnitude = 10 ** int(round(math.log10(range_val) - 1.0))
        range_normalized = range_val / magnitude
        
        if range_normalized < 1.5:
            step = 0.1 * magnitude
        elif range_normalized < 3:
            step = 0.2 * magnitude
        elif range_normalized < 7:
            step = 0.5 * magnitude
        else:
            step = 1.0 * magnitude
            
        return step
    
    def _generate_ticks(self, min_val: float, max_val: float, step: float) -> List[float]:
        """生成刻度值列表"""
        ticks = []
        current = math.ceil(min_val / step) * step
        while current <= max_val:
            ticks.append(current)
            current += step
        return ticks
    
    def _draw_data_series(self, x_values: List[float], y_values: List[float], 
                         x_min: float, x_max: float, y_min: float, y_max: float,
                         chart_width: int, chart_height: int, color: str):
        """绘制数据系列"""
        if len(x_values) != len(y_values) or len(x_values) == 0:
            return
        
        # 计算数据点坐标
        points = []
        for i in range(len(x_values)):
            x = self.margin + (x_values[i] - x_min) / (x_max - x_min) * chart_width
            y = self.canvas_size[1] - self.margin - (y_values[i] - y_min) / (y_max - y_min) * chart_height
            points.append((x, y))
        
        # 绘制折线
        self.draw.line(points, fill=color, width=2)
        
        # 绘制数据点
        for point in points:
            self.draw.ellipse([(point[0] - 3, point[1] - 3), (point[0] + 3, point[1] + 3)], 
                             fill=color)
    
    def _draw_legend(self, data_series: List[Dict[str, Any]]):
        """绘制图例"""
        legend_x = self.canvas_size[0] - self.margin - 120
        legend_y = self.margin + 20
        
        for i, series in enumerate(data_series):
            color = series.get('color', self.colors[i % len(self.colors)])
            y_pos = legend_y + i * 20
            
            # 绘制图例线条和点
            self.draw.line([(legend_x, y_pos), (legend_x + 20, y_pos)], fill=color, width=2)
            self.draw.ellipse([(legend_x + 10 - 3, y_pos - 3), (legend_x + 10 + 3, y_pos + 3)], 
                             fill=color)
            
            # 绘制图例文字
            self.draw.text((legend_x + 30, y_pos - 10), series['name'], fill='black')
    
    def _draw_title(self, title: str):
        """绘制标题"""
        self.draw.text((self.canvas_size[0] // 2 - len(title) * 4, self.margin - 30), 
                      title, fill='black')
    
    def save_image(self, file_path: str):
        """保存图表为图片"""
        # 确保目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        self.canvas.save(file_path)