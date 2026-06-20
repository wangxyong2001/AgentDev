"""
天气工具 — 模拟天气数据查询。

生产环境升级: 替换为真实天气 API（OpenWeatherMap 等）。
"""

from __future__ import annotations

from agentic.tools.registry import Tool

# ── 模拟天气数据库 ───────────────────────────────────────────────────

_WEATHER_DB = {
    "London": "Rainy, 12°C",
    "Tokyo": "Sunny, 25°C",
    "New York": "Cloudy, 18°C",
    "Paris": "Sunny, 20°C",
}


def _get_weather(city: str) -> str:
    """
    查询某个城市的天气。

    处理逻辑:
      1. 在模拟天气数据库中按城市名称查找
      2. 若找到，返回对应的天气描述字符串
      3. 若未找到，返回"未找到该城市的天气数据"

    参数:
      city: 城市名称（英文），例如 "London"、"Tokyo"

    返回值:
      天气描述字符串，例如 "Rainy, 12°C"
      或 "Weather data not found for this city."
    """
    return _WEATHER_DB.get(city, f"Weather data not found for this city.")


# ── 工具描述符 ─────────────────────────────────────────────────────

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city (e.g. 'London', 'Tokyo', 'New York', 'Paris')",
    func=_get_weather,
    metadata={"version": "1.0.0", "data_source": "simulated", "cities": len(_WEATHER_DB)},
)
