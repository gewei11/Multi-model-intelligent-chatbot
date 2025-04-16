from tools.weather_chart_tools import WeatherChartTool
import os

def main():
    # 创建图表工具实例
    chart_tool = WeatherChartTool()

    # 准备数据
    dates = ['2025-04-11', '2025-04-12', '2025-04-13']
    high_temps = [26, 21, 25]
    low_temps = [13, 10, 11]

    # 绘制图表
    chart_tool.draw_temperature_chart(
        dates=dates,
        high_temps=high_temps,
        low_temps=low_temps
    )

    # 保存图表
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'weather_chart.png')
    chart_tool.save_image(output_path)
    print(f'天气图表已保存至：{output_path}')

if __name__ == '__main__':
    main()