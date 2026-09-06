from typing import Annotated

import requests
from langchain_core.tools import tool, InjectedToolCallId


@tool
def get_weather(
        city: str,
        tool_call_id: Annotated[str, InjectedToolCallId], ):
    """获取指定城市的当前天气信息。

    当查询某一个城市的天气、湿度、温度时可以使用此工具

    Args:
        city: 城市名称，例如北京、上海

    Returns:
        {
            "province": "上海市",
            "city": "上海",
            "adcode": "310000",
            "weather": "多云",
            "weather_icon": "101",
            "temperature": 29,
            "wind_direction": "东北风",
            "wind_power": "2级",
            "humidity": 80,
            "report_time": "5 分钟前发布"
        }
        province: 省份
        city: 城市
        adcode: 城市编码
        weather: 天气
        weather_icon: 天气编码
        temperature: 温度
        wind_direction: 风向
        wind_power: 风力
        humidity: 湿度
        report_time: 发布时间

    """
    print(f'当前的tool id {tool_call_id}')


    res = requests.get(f'https://uapis.cn/api/v1/misc/weather?city={city}').json()
    return f"{city} 天气 {res['weather']} , {res['temperature']} 摄氏度"


if __name__ == "__main__":
    res = get_weather.invoke('北京')
    print(res)
