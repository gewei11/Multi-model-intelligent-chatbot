import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.sentiment_tools import SentimentAnalysisTool
from utils.config import load_config

def test_sentiment_analysis():
    """测试情感分析功能"""
    # 加载配置
    config = load_config()
    
    # 创建情感分析工具实例
    sentiment_tool = SentimentAnalysisTool(config)
    
    # 测试文本
    test_texts = [
        "今天天气真好，心情特别愉快！",
        "这个产品质量太差了，一点都不好用。",
        "这件事情还行吧，一般般。"
    ]
    
    # 分析每个测试文本
    for text in test_texts:
        print(f"\n分析文本: {text}")
        print("-" * 50)
        result = sentiment_tool.analyze_sentiment(text)
        print(f"分析结果: {result}")

def test_emotion_analysis():
    """测试对话情绪分析功能"""
    # 加载配置
    config = load_config()
    
    # 创建情感分析工具实例
    sentiment_tool = SentimentAnalysisTool(config)
    
    # 测试文本
    test_texts = [
        "你真是个大笨蛋！",
        "今天天气真好啊，我很开心！",
        "这件事情太让我失望了。",
        "对不起，我错了。",
        "我好害怕啊！"
    ]
    
    # 分析每个测试文本的情绪
    for text in test_texts:
        print(f"\n分析文本: {text}")
        print("-" * 50)
        result = sentiment_tool.analyze_emotion(text)
        if "error" not in result:
            print(f"主要情绪: {result['main_emotion']} (置信度: {result['confidence']:.2%})")
            print("\n详细情绪分析:")
            for emotion in result["detailed_emotions"]:
                print(f"- {emotion['type']}: {emotion['probability']:.2%}")
                if emotion['sub_emotions']:
                    print("  子情绪:")
                    for sub_emotion in emotion['sub_emotions']:
                        print(f"  - {sub_emotion['type']}: {sub_emotion['probability']:.2%}")
            if result["suggested_replies"]:
                print("\n建议回复:")
                for reply in result["suggested_replies"]:
                    print(f"- {reply}")
        else:
            print(f"分析失败: {result['error']}")

def test_combined_analysis():
    """测试组合分析功能（情感倾向+对话情绪）"""
    # 加载配置
    config = load_config()
    
    # 创建情感分析工具实例
    sentiment_tool = SentimentAnalysisTool(config)
    
    # 测试文本
    test_texts = [
        "你真是个大笨蛋！",
        "今天天气真好啊，我很开心！",
        "这个产品质量太差了，让人失望。",
        "谢谢你的帮助，你真是太好了！",
        "这件事情还行吧，一般般。"
    ]
    
    # 分析每个测试文本
    for text in test_texts:
        print(f"\n综合分析文本: {text}")
        print("-" * 50)
        
        result = sentiment_tool.analyze_combined(text)
        
        if "error" not in result:
            print("\n1. 情感倾向分析：")
            sentiment = result["sentiment_analysis"]
            print(f"情感倾向: {sentiment['sentiment']}")
            print(f"置信度: {sentiment['confidence']:.2%}")
            print(f"积极概率: {sentiment['positive_prob']:.2%}")
            print(f"消极概率: {sentiment['negative_prob']:.2%}")
            
            print("\n2. 对话情绪分析：")
            emotion = result["emotion_analysis"]
            print(f"主要情绪: {emotion['main_emotion']}")
            print(f"置信度: {emotion['confidence']:.2%}")
            
            print("\n详细情绪分布:")
            for e in emotion["detailed_emotions"]:
                print(f"- {e['type']}: {e['probability']:.2%}")
                if e['sub_emotions']:
                    print("  子情绪:")
                    for sub_e in e['sub_emotions']:
                        print(f"  - {sub_e['type']}: {sub_e['probability']:.2%}")
            
            if emotion["suggested_replies"]:
                print("\n建议回复:")
                for reply in emotion["suggested_replies"]:
                    print(f"- {reply}")
            
            print(f"\n综合结论：")
            print(result["conclusion"])
            
        else:
            print(f"分析失败: {result['error']}")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    print("1. 测试情感倾向分析...")
    test_sentiment_analysis()
    
    print("\n2. 测试对话情绪分析...")
    test_emotion_analysis()
    
    print("\n3. 测试综合分析...")
    test_combined_analysis()
