"""Weather via Open-Meteo (free, no key, no sign-up), so weather questions never depend on
web search or scraping weather sites."""

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

US_STATES = {
    "al": "alabama", "ak": "alaska", "az": "arizona", "ar": "arkansas", "ca": "california",
    "co": "colorado", "ct": "connecticut", "de": "delaware", "fl": "florida", "ga": "georgia",
    "hi": "hawaii", "id": "idaho", "il": "illinois", "in": "indiana", "ia": "iowa",
    "ks": "kansas", "ky": "kentucky", "la": "louisiana", "me": "maine", "md": "maryland",
    "ma": "massachusetts", "mi": "michigan", "mn": "minnesota", "ms": "mississippi",
    "mo": "missouri", "mt": "montana", "ne": "nebraska", "nv": "nevada", "nh": "new hampshire",
    "nj": "new jersey", "nm": "new mexico", "ny": "new york", "nc": "north carolina",
    "nd": "north dakota", "oh": "ohio", "ok": "oklahoma", "or": "oregon", "pa": "pennsylvania",
    "ri": "rhode island", "sc": "south carolina", "sd": "south dakota", "tn": "tennessee",
    "tx": "texas", "ut": "utah", "vt": "vermont", "va": "virginia", "wa": "washington",
    "wv": "west virginia", "wi": "wisconsin", "wy": "wyoming", "dc": "district of columbia",
}  # fmt: skip

# WMO weather interpretation codes.
CONDITIONS = {
    0: "Clear", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog",
    48: "Freezing fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Heavy freezing drizzle", 61: "Light rain", 63: "Rain",
    65: "Heavy rain", 66: "Freezing rain", 67: "Heavy freezing rain", 71: "Light snow",
    73: "Snow", 75: "Heavy snow", 77: "Snow grains", 80: "Light showers", 81: "Showers",
    82: "Violent showers", 85: "Snow showers", 86: "Heavy snow showers", 95: "Thunderstorm",
    96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}  # fmt: skip

IMPERIAL_COUNTRIES = {"US", "LR", "MM"}


class WeatherError(Exception):
    pass


def _matches(place: dict, qualifiers: list[str]) -> bool:
    fields = {
        str(place.get(k, "")).lower()
        for k in ("admin1", "admin2", "country", "country_code")
        if place.get(k)
    }
    for q in qualifiers:
        q = US_STATES.get(q, q)
        if q in {"usa", "us", "united states"}:
            q = "united states"
        if not any(q == f or q in f for f in fields):
            return False
    return True


async def geocode(location: str, client: httpx.AsyncClient) -> dict:
    """Resolve "Redmond, WA", "Paris, France" or "98052"-style input to one place.

    Street addresses are fine: only the city part is used (the first part that isn't a
    number or street).
    """
    parts = [p.strip() for p in location.split(",") if p.strip()]
    if not parts:
        raise WeatherError("No location given")
    # "123 Main St, Redmond, WA 98052": drop leading street parts and trailing zip codes.
    if len(parts) >= 3 and any(c.isdigit() for c in parts[0]):
        parts = parts[1:]
    parts = [" ".join(w for w in p.split() if not w.isdigit()) or p for p in parts]
    name, qualifiers = parts[0], [p.lower() for p in parts[1:]]
    resp = await client.get(
        GEOCODE_URL, params={"name": name, "count": 10, "format": "json"}, timeout=15
    )
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        raise WeatherError(f"Couldn't find a place called '{name}'")
    return next((r for r in results if _matches(r, qualifiers)), results[0])


def _place_name(p: dict) -> str:
    return ", ".join(x for x in (p.get("name"), p.get("admin1"), p.get("country")) if x)


async def forecast(location: str, client: httpx.AsyncClient, days: int = 3) -> str:
    place = await geocode(location, client)
    imperial = place.get("country_code") in IMPERIAL_COUNTRIES
    t_unit, w_unit, p_unit = ("°F", "mph", "in") if imperial else ("°C", "km/h", "mm")
    resp = await client.get(
        FORECAST_URL,
        params={
            "latitude": place["latitude"],
            "longitude": place["longitude"],
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "weather_code,wind_speed_10m,precipitation",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": max(1, min(days, 16)),
            "temperature_unit": "fahrenheit" if imperial else "celsius",
            "wind_speed_unit": "mph" if imperial else "kmh",
            "precipitation_unit": "inch" if imperial else "mm",
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    cur, daily = data.get("current", {}), data.get("daily", {})
    lines = [f"Weather for {_place_name(place)} (local time {cur.get('time', '?')}):"]
    if cur:
        lines.append(
            f"Now: {CONDITIONS.get(cur.get('weather_code'), 'Unknown')}, "
            f"{cur.get('temperature_2m')}{t_unit} (feels like {cur.get('apparent_temperature')}"
            f"{t_unit}), humidity {cur.get('relative_humidity_2m')}%, wind "
            f"{cur.get('wind_speed_10m')} {w_unit}, "
            f"precipitation {cur.get('precipitation')} {p_unit}"
        )
    for i, day in enumerate(daily.get("time", [])):
        lines.append(
            f"{day}: {CONDITIONS.get(daily['weather_code'][i], 'Unknown')}, "
            f"high {daily['temperature_2m_max'][i]}{t_unit}, low {daily['temperature_2m_min'][i]}"
            f"{t_unit}, {daily['precipitation_probability_max'][i]}% chance of precipitation "
            f"({daily['precipitation_sum'][i]} {p_unit}), wind up to "
            f"{daily['wind_speed_10m_max'][i]} {w_unit}"
        )
    lines.append("Source: Open-Meteo (https://open-meteo.com)")
    return "\n".join(lines)
