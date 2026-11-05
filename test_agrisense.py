"""Tests for the climate lookup, weather helper and Streamlit app.

All network calls are mocked, so these run offline:
    pip install pytest
    pytest -q
"""
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import climate  # noqa: E402
import predict  # noqa: E402
import weather  # noqa: E402

MONTHS = climate.MONTH_KEYS


def fake(status=200, payload=None):
    r = mock.Mock()
    r.status_code = status
    r.json.return_value = payload if payload is not None else {}
    return r


def power_payload(temp=28.0, rh=70.0, rain_per_day=5.0, **overrides):
    """Shape of a NASA POWER climatology response (monthly keys + ANN)."""
    def block(v):
        d = {m: v for m in MONTHS}
        d["ANN"] = v
        return d
    params = {"T2M": block(temp), "RH2M": block(rh), "PRECTOTCORR": block(rain_per_day)}
    params.update(overrides)
    return {"properties": {"parameter": params}}


GEO = {"results": [
    {"name": "Lucknow", "latitude": 26.84, "longitude": 80.92,
     "country": "India", "country_code": "IN", "admin1": "Uttar Pradesh"},
]}


def router(power=None, geo=None):
    """requests.get stand-in that answers geocoding and NASA POWER calls."""
    def _get(url, params=None, timeout=None):
        if "geocoding" in url:
            return geo or fake(200, GEO)
        if "power.larc.nasa.gov" in url:
            return power or fake(200, power_payload())
        raise AssertionError(f"unexpected URL {url}")
    return _get


# ------------------------------------------------------------------ climate
def test_seasonal_average_and_rain_conversion():
    # 5 mm/day -> 5 * days in month; June (30 days) = 150 mm
    with mock.patch("requests.get", side_effect=router()):
        out = climate.get_seasonal_climate(26.8, 80.9, [6])
    assert out["temperature"] == 28.0
    assert out["humidity"] == 70.0
    assert out["rainfall"] == 150.0


def test_season_uses_each_months_length():
    with mock.patch("requests.get", side_effect=router()):
        out = climate.get_seasonal_climate(26.8, 80.9, [1, 2])  # 31 and 28.25 days
    assert out["rainfall"] == pytest.approx((5 * 31 + 5 * 28.25) / 2, abs=0.1)
    assert [m["Month"] for m in out["monthly"]] == ["Jan", "Feb"]


def test_city_flow_returns_place_name():
    with mock.patch("requests.get", side_effect=router()):
        out = climate.climate_for_city("Lucknow,IN", [6, 7])
    assert out["place"] == "Lucknow, Uttar Pradesh, India"
    assert out["latitude"] == pytest.approx(26.84)


def test_geocode_country_hint_filters_results():
    two = {"results": [
        {"name": "Springfield", "latitude": 1, "longitude": 1, "country": "United States",
         "country_code": "US", "admin1": "Illinois"},
        {"name": "Springfield", "latitude": 2, "longitude": 2, "country": "Australia",
         "country_code": "AU", "admin1": "Queensland"},
    ]}
    with mock.patch("requests.get", return_value=fake(200, two)):
        assert climate.geocode("Springfield,AU")["latitude"] == 2.0
        with pytest.raises(climate.ClimateError, match="Could not find"):
            climate.geocode("Springfield,IN")


@pytest.mark.parametrize("place", ["", "   ", None])
def test_geocode_rejects_empty_city(place):
    with pytest.raises(climate.ClimateError, match="city name"):
        climate.geocode(place)


def test_unknown_city():
    with mock.patch("requests.get", return_value=fake(200, {})):
        with pytest.raises(climate.ClimateError, match="Could not find"):
            climate.geocode("Nowhereville")


def test_input_validation():
    for months in ([], [0], [13], ["Jun"]):
        with pytest.raises(climate.ClimateError):
            climate.get_seasonal_climate(26.8, 80.9, months)
    with pytest.raises(climate.ClimateError, match="coordinates"):
        climate.get_seasonal_climate(120, 80.9, [6])


def test_nasa_missing_value_over_ocean():
    with mock.patch("requests.get", side_effect=router(power=fake(200, power_payload(temp=-999)))):
        with pytest.raises(climate.ClimateError, match="no data"):
            climate.get_seasonal_climate(0, -30, [6])


def test_nasa_http_error_and_bad_format_and_network():
    with mock.patch("requests.get", side_effect=router(power=fake(500))):
        with pytest.raises(climate.ClimateError, match="HTTP 500"):
            climate.get_seasonal_climate(26.8, 80.9, [6])
    with mock.patch("requests.get", side_effect=router(power=fake(200, {"oops": 1}))):
        with pytest.raises(climate.ClimateError, match="unexpected format"):
            climate.get_seasonal_climate(26.8, 80.9, [6])
    with mock.patch("requests.get", side_effect=requests.ConnectionError("down")):
        with pytest.raises(climate.ClimateError, match="Could not reach"):
            climate.get_seasonal_climate(26.8, 80.9, [6])


# ------------------------------------------------------------------ OpenWeatherMap
OWM = {"name": "Lucknow", "sys": {"country": "IN"}, "main": {"temp": 29.4, "humidity": 62},
       "weather": [{"description": "haze"}]}


def test_openweather_success_and_params():
    with mock.patch("requests.get", return_value=fake(200, OWM)) as g:
        out = weather.get_current_weather("Lucknow,IN", api_key="abc")
    assert out["temperature"] == 29.4 and out["humidity"] == 62.0
    assert g.call_args.kwargs["params"] == {"q": "Lucknow,IN", "appid": "abc", "units": "metric"}


@pytest.mark.parametrize("status,msg", [(401, "rejected"), (404, "not found"), (500, "HTTP 500")])
def test_openweather_errors(status, msg):
    with mock.patch("requests.get", return_value=fake(status)):
        with pytest.raises(weather.WeatherError, match=msg):
            weather.get_current_weather("X", api_key="abc")


def test_openweather_missing_key_and_empty_city(monkeypatch):
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    with pytest.raises(weather.WeatherError, match="API key"):
        weather.get_current_weather("Lucknow,IN")
    with pytest.raises(weather.WeatherError, match="city name"):
        weather.get_current_weather("  ", api_key="abc")


# ------------------------------------------------------------------ model
def test_model_recommends_known_crop():
    # Typical rice conditions from the dataset (first row).
    inputs = dict(N=90, P=42, K=43, temperature=21, humidity=82, ph=6.5, rainfall=203)
    top = predict.recommend(inputs, top_k=3)
    assert len(top) == 3
    assert top[0][0] == "rice"
    assert top[0][1] > top[1][1] >= top[2][1]
    assert sum(p for _, p in top) <= 1.0 + 1e-9


# ------------------------------------------------------------------ app
@pytest.fixture
def app():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=60)
    return at.run()


def click(at, key):
    return next(b for b in at.button if b.key == key).click().run()


def sliders(at):
    return {s.label: s.value for s in at.slider}


def test_app_loads_cleanly(app):
    assert not app.exception


def test_app_fills_sliders_from_nasa(app):
    app.text_input(key="city").set_value("Lucknow,IN").run()
    with mock.patch("requests.get", side_effect=router()):
        click(app, "fetch_climate")
    assert not app.exception
    s = sliders(app)
    assert s["Temperature (°C)"] == 28.0
    assert s["Humidity (%)"] == 70.0
    # Kharif = Jun-Oct at 5 mm/day: average month length is 30.6 days -> ~153 mm/month
    assert s["Rainfall (mm per month)"] == pytest.approx(5 * 30.6, abs=1)
    assert not app.error


def test_app_clamps_out_of_range_rainfall_and_says_so(app):
    app.text_input(key="city").set_value("Cherrapunji,IN").run()
    wet = fake(200, power_payload(rain_per_day=15.0))  # ~450 mm/month > slider max 300
    with mock.patch("requests.get", side_effect=router(power=wet)):
        click(app, "fetch_climate")
    assert not app.exception
    assert sliders(app)["Rainfall (mm per month)"] == 300.0
    assert any("outside the slider range" in i.value for i in app.info)


def test_app_shows_error_for_unknown_city(app):
    app.text_input(key="city").set_value("Nowhereville").run()
    with mock.patch("requests.get", return_value=fake(200, {})):
        click(app, "fetch_climate")
    assert any("Could not find" in e.value for e in app.error)


def test_app_openweather_button(app, monkeypatch):
    monkeypatch.setenv("OPENWEATHER_API_KEY", "abc")
    app.text_input(key="owm_city").set_value("Lucknow,IN").run()
    with mock.patch("requests.get", return_value=fake(200, OWM)):
        click(app, "fetch_owm")
    assert not app.exception
    s = sliders(app)
    assert abs(s["Temperature (°C)"] - 29.4) < 0.6 and s["Humidity (%)"] == 62.0


def test_app_end_to_end_recommendation(app):
    app.text_input(key="city").set_value("Lucknow,IN").run()
    with mock.patch("requests.get", side_effect=router()):
        click(app, "fetch_climate")
    next(b for b in app.button if b.label == "Get recommendation").click().run()
    assert not app.exception
    assert any("Recommended crop" in s.value for s in app.success)
