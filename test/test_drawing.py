from agents.drawing_agent import DrawingAgent

def test_draw_line():
    # 创建配置字典
    config = {
        "models": {
            "deepseek": {
                "api_base": "http://localhost:11434/api/chat",
                "model_name": "MFDoom/deepseek-r1-tool-calling:8b",
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "qwen": {
                "api_base": "http://localhost:11434/api/chat",
                "model_name": "qwen2.5:7b",
                "max_tokens": 2048,
                "temperature": 0.7
            }
        },
        "logging": {
            "file": "logs/chatbot.log",
            "level": "INFO"
        }
    }
    
    # 创建绘图代理实例
    agent = DrawingAgent(config)
    
    # 测试绘制红色线条
    result = agent.process_message('画一条红色线条从(100,100)到(200,200)')
    print(result)

if __name__ == '__main__':
    test_draw_line()