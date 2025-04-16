from typing import Dict, Any
from dataclasses import dataclass, field
import re

@dataclass
class WeatherQueryTool:
    """天气查询工具"""
    name: str = "weather_query"
    description: str = "查询指定城市的天气信息，支持实时天气和未来天气预报"
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "要查询的城市名称"
            },
            "days": {
                "type": "integer",
                "description": "要查询的天数，1表示今天，2表示今明两天，最多3天",
                "minimum": 1,
                "maximum": 3
            }
        },
        "required": ["city", "days"]
    })

    def parse_query(self, text: str) -> Dict[str, Any]:
        """解析用户查询意图"""
        # 确定查询天数
        days = 1  # 默认查询今天
        if "明天" in text:
            days = 2
        elif "后天" in text:
            days = 3
        elif "未来" in text:
            # 匹配数字和"天"
            match = re.search(r'未来\s*(\d+)\s*天', text)
            if match:
                days = min(int(match.group(1)), 3)
            elif "一周" in text or "7天" in text:
                days = 3
            else:
                # 默认未来3天
                days = 3
        
        # 移除时间相关词汇，避免干扰城市名提取
        clean_text = re.sub(r'(天气|查询|的|怎么样|如何|未来|今天|明天|后天|\d+天|一周)', '', text).strip()
        
        # 提取城市名（2-4个汉字）
        city_match = re.search(r'([一-龥]{2,4})', clean_text)
        city = city_match.group(1) if city_match else None
        
        return {
            "city": city,
            "days": days
        }
