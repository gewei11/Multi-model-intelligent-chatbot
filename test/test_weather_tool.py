from tools.weather_tools import WeatherQueryTool

def test_weather_query():
    tool = WeatherQueryTool()
    
    test_cases = [
        "成都未来一周的天气",
        "查询北京后天天气",
        "上海明天天气怎么样",
        "查询包头市天气",
        "成都未来5天天气"
    ]
    
    for query in test_cases:
        print(f"测试查询: {query}")
        result = tool.parse_query(query)
        print(f"解析结果: {result}\n")

if __name__ == "__main__":
    test_weather_query()