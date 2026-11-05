"""OpenWeatherMap helper for Agrisense.

Setup:
  1. Create a free account at https://openweathermap.org and copy your key
     from the "API keys" tab (a new key can take up to a couple of hours to work).
  2. Make the key available to the app, never in the code or on GitHub:
       - environment variable:  OPENWEATHER_API_KEY=your_key
       - or Streamlit secrets:  .streamlit/secrets.toml  ->  OPENWEATHER_API_KEY = "your_key"

CLI test:
    python src/weather.py "Lucknow,IN"
"""
import os
import sys

import requests

CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"


class WeatherError(Exception):
    """Raised with a user-friendly message when weather data can't be fetched."""


def get_api_key(explicit_key=None):
    key = explicit_key or os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise WeatherError(
            "No OpenWeatherMap API key found. Set the OPENWEATHER_API_KEY "
            "environment variable or add it to .streamlit/secrets.toml."
        )
    return key


def get_current_weather(city, api_key=None, timeout=10):
    """Return current temperature (C), humidity (%) and a short description for a city.

    Use "City,CountryCode" (e.g. "Lucknow,IN") to avoid picking the wrong city.
    """
    city = (city or "").strip()
    if not city:
        raise WeatherError("Please enter a city name.")

    key = get_api_key(api_key)
    try:
        resp = requests.get(
            CURRENT_URL,
            params={"q": city, "appid": key, "units": "metric"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise WeatherError(f"Could not reach OpenWeatherMap: {exc}") from exc

    if resp.status_code == 401:
        raise WeatherError(
            "OpenWeatherMap rejected the API key. Check it, and note that "
            "new keys can take up to a couple of hours to activate."
        )
    if resp.status_code == 404:
        raise WeatherError(f"City '{city}' was not found. Try 'City,CountryCode'.")
    if resp.status_code != 200:
        raise WeatherError(f"OpenWeatherMap returned an error (HTTP {resp.status_code}).")

    data = resp.json()
    return {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country", ""),
        "temperature": float(data["main"]["temp"]),
        "humidity": float(data["main"]["humidity"]),
        "description": data["weather"][0]["description"],
        # Only present while it is raining; this is the last hour, NOT a seasonal total.
        "rain_last_hour_mm": float(data.get("rain", {}).get("1h", 0.0)),
    }


if __name__ == "__main__":
    place = sys.argv[1] if len(sys.argv) > 1 else "Lucknow,IN"
    try:
        print(get_current_weather(place))
    except WeatherError as err:
        sys.exit(str(err))
