import asyncio

import httpx

from app.tools import weather

GEO = {
    "results": [
        {"name": "Redmond", "admin1": "Oregon", "country": "United States", "country_code": "US",
         "latitude": 44.27, "longitude": -121.17},
        {"name": "Redmond", "admin1": "Washington", "country": "United States",
         "country_code": "US", "latitude": 47.67, "longitude": -122.12},
    ]
}  # fmt: skip
FORECAST = {
    "current": {"time": "2026-10-11T15:00", "temperature_2m": 58.1, "apparent_temperature": 56.0,
                "relative_humidity_2m": 70, "weather_code": 3, "wind_speed_10m": 5.2,
                "precipitation": 0.0},
    "daily": {"time": ["2026-10-11"], "weather_code": [61], "temperature_2m_max": [60.0],
              "temperature_2m_min": [47.5], "precipitation_probability_max": [80],
              "precipitation_sum": [0.2], "wind_speed_10m_max": [9.1]},
}  # fmt: skip


def run(location, seen):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=GEO if "geocoding" in request.url.host else FORECAST)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await weather.forecast(location, client)

    return asyncio.run(go())


def test_state_abbreviation_picks_the_right_city():
    seen = []
    text = run("Redmond, WA", seen)
    assert seen[0].url.params["name"] == "Redmond"
    assert seen[1].url.params["latitude"] == "47.67"
    assert seen[1].url.params["temperature_unit"] == "fahrenheit"
    assert "Redmond, Washington, United States" in text
    assert "Overcast, 58.1°F" in text and "80% chance" in text and "Open-Meteo" in text


def test_street_address_uses_the_city():
    seen = []
    run("123 Main St, Redmond, WA 98052", seen)
    assert seen[0].url.params["name"] == "Redmond"
    assert seen[1].url.params["latitude"] == "47.67"
