"""Seasonal climate lookup for Agrisense. No API keys needed.

  1. Turn a city name into coordinates (Open-Meteo geocoding).
  2. Fetch long-term monthly climate for those coordinates (NASA POWER climatology).
  3. Average the months of a growing season into the three weather inputs the model uses.

Assumption: the training dataset's `rainfall` column is treated as AVERAGE MONTHLY
rainfall in mm. The dataset does not document its period, so this is an interpretation.
NASA POWER reports rainfall in mm/day, so each month is multiplied by its number of days.

CLI:
    python src/climate.py "Lucknow,IN" kharif
"""
import sys

import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
POWER_URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_KEYS = [m.upper() for m in MONTH_NAMES]
DAYS_IN_MONTH = [31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Approximate Indian cropping seasons (month numbers, in season order).
SEASONS = {
    "Kharif (Jun-Oct)": [6, 7, 8, 9, 10],
    "Rabi (Nov-Mar)": [11, 12, 1, 2, 3],
    "Zaid (Mar-Jun)": [3, 4, 5, 6],
    "Whole year": list(range(1, 13)),
}

MISSING_THRESHOLD = -990  # NASA POWER marks missing values as -999


class ClimateError(Exception):
    """Raised with a user-friendly message when climate data can't be fetched."""


def _get(url, params, timeout, what):
    try:
        return requests.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise ClimateError(f"Could not reach {what}: {exc}") from exc


def geocode(place, timeout=10):
    """City name -> coordinates. Accepts 'City' or 'City,CountryCode' (e.g. 'Lucknow,IN')."""
    parts = [p.strip() for p in (place or "").split(",") if p.strip()]
    if not parts:
        raise ClimateError("Please enter a city name.")
    name = parts[0]
    hint = parts[1].lower() if len(parts) > 1 else None

    resp = _get(
        GEOCODE_URL,
        {"name": name, "count": 10, "language": "en", "format": "json"},
        timeout,
        "the geocoding service",
    )
    if resp.status_code != 200:
        raise ClimateError(f"Geocoding service returned an error (HTTP {resp.status_code}).")

    results = resp.json().get("results") or []
    if hint:
        results = [
            r for r in results
            if hint in {
                str(r.get("country_code", "")).lower(),
                str(r.get("country", "")).lower(),
                str(r.get("admin1", "")).lower(),
            }
        ]
    if not results:
        raise ClimateError(
            f"Could not find '{place}'. Try a nearby larger city, or use "
            "'City,CountryCode' such as 'Lucknow,IN'."
        )
    top = results[0]
    return {
        "name": top.get("name", name),
        "admin1": top.get("admin1", ""),
        "country": top.get("country", ""),
        "latitude": float(top["latitude"]),
        "longitude": float(top["longitude"]),
    }


def get_seasonal_climate(latitude, longitude, months, timeout=30):
    """Average temperature (C), humidity (%) and rainfall (mm per month) over `months`."""
    months = list(dict.fromkeys(months))  # de-duplicate, keep order
    if not months:
        raise ClimateError("Pick at least one month.")
    if any((not isinstance(m, int)) or not 1 <= m <= 12 for m in months):
        raise ClimateError("Months must be numbers from 1 to 12.")
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        raise ClimateError("Invalid coordinates.")

    resp = _get(
        POWER_URL,
        {
            "parameters": "T2M,RH2M,PRECTOTCORR",
            "community": "AG",
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "format": "JSON",
        },
        timeout,
        "NASA POWER",
    )
    if resp.status_code != 200:
        raise ClimateError(
            f"NASA POWER returned an error (HTTP {resp.status_code}). "
            "Check the location and try again."
        )

    try:
        params = resp.json()["properties"]["parameter"]
        t2m, rh2m, prec = params["T2M"], params["RH2M"], params["PRECTOTCORR"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ClimateError("NASA POWER returned data in an unexpected format.") from exc

    monthly = []
    for m in months:
        key = MONTH_KEYS[m - 1]
        try:
            t, h, p = float(t2m[key]), float(rh2m[key]), float(prec[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ClimateError(f"NASA POWER data is missing for {MONTH_NAMES[m - 1]}.") from exc
        if min(t, h, p) <= MISSING_THRESHOLD:
            raise ClimateError(
                "NASA POWER has no data for this location and month (for example over open ocean)."
            )
        monthly.append({
            "Month": MONTH_NAMES[m - 1],
            "Temperature (°C)": round(t, 1),
            "Humidity (%)": round(h, 1),
            "Rainfall (mm)": round(p * DAYS_IN_MONTH[m - 1], 1),
        })

    n = len(monthly)
    return {
        "temperature": sum(r["Temperature (°C)"] for r in monthly) / n,
        "humidity": sum(r["Humidity (%)"] for r in monthly) / n,
        "rainfall": sum(r["Rainfall (mm)"] for r in monthly) / n,
        "monthly": monthly,
    }


def climate_for_city(place, months):
    """City name + months -> season-average climate, plus the resolved place name."""
    loc = geocode(place)
    data = get_seasonal_climate(loc["latitude"], loc["longitude"], months)
    data["place"] = ", ".join(x for x in (loc["name"], loc["admin1"], loc["country"]) if x)
    data["latitude"], data["longitude"] = loc["latitude"], loc["longitude"]
    return data


if __name__ == "__main__":
    city = sys.argv[1] if len(sys.argv) > 1 else "Lucknow,IN"
    wanted = sys.argv[2].lower() if len(sys.argv) > 2 else "kharif"
    season = next((k for k in SEASONS if k.lower().startswith(wanted)), "Kharif (Jun-Oct)")
    try:
        result = climate_for_city(city, SEASONS[season])
    except ClimateError as err:
        sys.exit(str(err))
    print(f"{result['place']}  |  {season}")
    print(f"  temperature: {result['temperature']:.1f} C")
    print(f"  humidity:    {result['humidity']:.0f} %")
    print(f"  rainfall:    {result['rainfall']:.0f} mm/month")
