import json
import requests
import re
from typing import Dict, Any, List, Optional
from pypinyin import lazy_pinyin

from utils.logger import get_logger
from utils.helper_functions import retry, safe_json_loads
from utils.config import get_api_key
from tools.weather_tools import WeatherQueryTool

class WeatherAgent:
    def __init__(self, config: Dict[str, Any]):
        """初始化天气Agent"""
        self.config = config
        self.logger = get_logger("weather_agent")
        self.logger.info("天气Agent初始化")
        
        # 获取API配置
        self.api_key = get_api_key("weather", config)
        self.base_url = config.get("apis", {}).get("weather", {}).get("base_url")
        self.forecast_url = config.get("apis", {}).get("weather", {}).get("forecast_url")
        
        # 城市名缓存
        self.city_name_cache = {}
        
        # 初始化查询工具
        self.query_tool = WeatherQueryTool()

    def process(self, user_input: str, context: Dict[str, Any]) -> str:
        """处理天气查询请求"""
        self.logger.info(f"处理天气查询: {user_input}")
        
        try:
            # 使用工具解析查询
            params = self.query_tool.parse_query(user_input)
            self.logger.info(f"解析参数: {params}")
            
            if not params["city"]:
                return "抱歉，我没有识别出您想查询哪个城市的天气。请明确指定城市名称，例如'北京今天天气怎么样？'"
            
            # 获取天气数据
            weather_data = self._get_weather_data(params["city"], params["days"])
            
            if "error" in weather_data:
                return f"抱歉，获取{params['city']}的天气信息失败：{weather_data['error']}"
            
            # 格式化响应
            return self._format_weather_response(weather_data, params["days"])
            
        except Exception as e:
            self.logger.error(f"处理查询失败: {str(e)}")
            return f"抱歉，处理天气查询时发生错误。请稍后重试。"

    def _translate_city_name(self, chinese_name: str) -> str:
        """将中文城市名转换为英文"""
        # 检查缓存
        if chinese_name in self.city_name_cache:
            return self.city_name_cache[chinese_name]
        
        try:
            # 清理城市名（移除"市县区"等后缀）
            clean_name = chinese_name.rstrip('市县区')
            
            # 使用拼音作为城市名
            from pypinyin import lazy_pinyin
            pinyin = ''.join(lazy_pinyin(clean_name))
            pinyin = pinyin.capitalize()  # 首字母大写
            
            self.logger.info(f"城市名转换: {clean_name} -> {pinyin}")
            self.city_name_cache[chinese_name] = pinyin
            return pinyin
            
        except Exception as e:
            self.logger.error(f"城市名转换失败: {str(e)}")
            return chinese_name

    def get_weather(self, loc: str, days: int = 1) -> Dict[str, Any]:
        """获取城市天气信息和预报"""
        self.logger.info(f"获取{loc}的天气数据，预报天数：{days}")
        
        try:
            if days > 1:
                # 获取天气预报
                forecast_data = self._get_forecast_weather(loc, days)
                if "error" in forecast_data:
                    return forecast_data
                
                # 添加实时天气数据
                now_data = self._get_now_weather(loc)
                if "error" not in now_data:
                    forecast_data["now"] = now_data["now"]
                
                return forecast_data
            else:
                # 只获取实时天气
                return self._get_now_weather(loc)
            
        except Exception as e:
            self.logger.error(f"获取天气数据失败: {str(e)}")
            return {"error": f"获取天气数据失败: {str(e)}"}

    def _get_weather_data(self, city: str, days: int) -> Dict[str, Any]:
        """获取天气数据"""
        self.logger.info(f"获取天气数据: 城市={city}, 天数={days}")
        
        try:
            # 转换城市名为拼音
            from pypinyin import lazy_pinyin
            city_pinyin = ''.join(lazy_pinyin(city))
            self.logger.info(f"城市名转换: {city} -> {city_pinyin}")
            
            # 获取天气数据
            if days > 1:
                # 获取天气预报
                params = {
                    "key": self.api_key,
                    "location": city_pinyin,
                    "language": "zh-Hans",
                    "unit": "c",
                    "days": days
                }
                url = "https://api.seniverse.com/v3/weather/daily.json"
            else:
                # 获取实时天气
                params = {
                    "key": self.api_key,
                    "location": city_pinyin,
                    "language": "zh-Hans",
                    "unit": "c"
                }
                url = "https://api.seniverse.com/v3/weather/now.json"
            
            # 发送请求
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                result = data["results"][0]
                if days > 1:
                    # 补充实时天气数据
                    now_data = self._get_now_weather(city_pinyin)
                    if "error" not in now_data:
                        result["now"] = now_data["now"]
                return result
            
            return {"error": "未获取到天气数据"}
            
        except Exception as e:
            self.logger.error(f"获取天气数据失败: {str(e)}")
            return {"error": f"获取天气数据失败: {str(e)}"}

    def _get_now_weather(self, loc: str) -> Dict[str, Any]:
        """获取实时天气数据"""
        params = {
            "key": self.api_key,
            "location": loc,
            "language": "zh-Hans",
            "unit": "c"
        }
        
        try:
            url = "https://api.seniverse.com/v3/weather/now.json"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]
            return {"error": "未获取到实时天气数据"}
            
        except Exception as e:
            return {"error": f"获取实时天气失败: {str(e)}"}

    def _get_forecast_weather(self, loc: str, days: int = 5) -> Dict[str, Any]:
        """获取天气预报数据"""
        params = {
            "key": self.api_key,
            "location": loc,
            "language": "zh-Hans",
            "unit": "c",
            "start": 0,
            "days": min(days, 15)  # 限制最多15天
        }

        try:
            url = "https://api.seniverse.com/v3/weather/daily.json"
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if "results" in data and len(data["results"]) > 0:
                return data["results"][0]
            return {"error": "未获取到天气预报数据"}
            
        except Exception as e:
            return {"error": f"获取天气预报失败: {str(e)}"}

    def _format_weather_response(self, weather_data: Dict[str, Any], days: int) -> str:
        """格式化天气信息响应"""
        try:
            location = weather_data["location"]["name"]
            response = [f"📍 {location}"]
            
            # 添加实时天气
            if "now" in weather_data:
                now = weather_data["now"]
                response.extend([
                    "\n🕐 当前天气：",
                    f"🌡️ 温度：{now['temperature']}°C",
                    f"🌤️ 天气：{now['text']}"
                ])
            
            # 添加预报信息
            if "daily" in weather_data and days > 1:
                response.append("\n📅 天气预报：")
                for day in weather_data["daily"][:days]:
                    response.extend([
                        f"\n{day['date']}：",
                        f"🌡️ 温度：{day['low']}°C ~ {day['high']}°C",
                        f"🌅 白天：{day['text_day']}",
                        f"🌙 夜间：{day['text_night']}",
                        f"🌬️ 风向：{day['wind_direction']} 风力：{day['wind_scale']}级",
                        f"💧 相对湿度：{day['humidity']}%"
                    ])
            
            response.append(f"\n🔄 更新时间：{weather_data['last_update']}")
            return "\n".join(response)
            
        except Exception as e:
            self.logger.error(f"格式化天气数据失败: {str(e)}")
            return "抱歉，天气信息格式化失败。请稍后重试。"