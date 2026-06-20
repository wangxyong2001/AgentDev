"""
Weather tool — simulated weather data lookup.

Production upgrade: replace with real weather API (OpenWeatherMap, etc.).
"""

from __future__ import annotations

from agentic.tools.registry import Tool

# ── Simulated weather database ───────────────────────────────────────

_WEATHER_DB = {
    "London": "Rainy, 12°C",
    "Tokyo": "Sunny, 25°C",
    "New York": "Cloudy, 18°C",
    "Paris": "Sunny, 20°C",
}


def _get_weather(city: str) -> str:
    """
    Get weather for a city.

    Args:
      city: City name (English), e.g. "London", "Tokyo"

    Returns:
      Weather description string, e.g. "Rainy, 12°C"
      or "Weather data not found for this city."
    """
    return _WEATHER_DB.get(city, f"Weather data not found for this city.")


# ── Tool descriptor ─────────────────────────────────────────────────

weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city (e.g. 'London', 'Tokyo', 'New York', 'Paris')",
    func=_get_weather,
    metadata={"version": "1.0.0", "data_source": "simulated", "cities": len(_WEATHER_DB)},
)
