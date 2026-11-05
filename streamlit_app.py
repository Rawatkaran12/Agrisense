"""Agrisense web app. Run with:  streamlit run app/streamlit_app.py"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from climate import MONTH_NAMES, SEASONS, ClimateError, climate_for_city  # noqa: E402
from predict import load_model, recommend  # noqa: E402
from weather import WeatherError, get_current_weather  # noqa: E402

st.set_page_config(page_title="Agrisense", page_icon="🌾", layout="centered")
st.title("🌾 Agrisense")
st.caption("Crop recommendation from soil and weather conditions")

# (min, max, step) of the sliders; fetched values are snapped into these.
LIMITS = {
    "temperature": (5.0, 45.0, 0.5),
    "humidity": (10.0, 100.0, 1.0),
    "rainfall": (15.0, 300.0, 1.0),
}
LABELS = {"temperature": "Temperature", "humidity": "Humidity", "rainfall": "Rainfall"}


@st.cache_resource
def get_model():
    return load_model()


def get_weather_key():
    """OpenWeatherMap key from the environment or Streamlit secrets (never hardcode it)."""
    key = os.getenv("OPENWEATHER_API_KEY")
    if key:
        return key
    try:
        return st.secrets["OPENWEATHER_API_KEY"]
    except Exception:
        return None


def apply_to_sliders(values):
    """Put fetched values into the sliders. Returns notes for any value that had to be clamped."""
    notes = []
    for name, value in values.items():
        low, high, step = LIMITS[name]
        snapped = round(round(value / step) * step, 2)
        clamped = float(min(max(snapped, low), high))
        if clamped != snapped:
            notes.append(f"{LABELS[name]} ({value:.0f}) is outside the slider range "
                         f"and was set to {clamped:g}.")
        st.session_state[name] = clamped
    return notes


try:
    model, features = get_model()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Slider values live in session state so the fetch buttons can update them.
st.session_state.setdefault("temperature", 24.0)
st.session_state.setdefault("humidity", 80.0)
st.session_state.setdefault("rainfall", 200.0)
st.session_state.setdefault("climate_result", None)
st.session_state.setdefault("climate_error", None)

# ---------------------------------------------------------------- location + season
st.subheader("📍 Location and season")
st.caption("Fills temperature, humidity and rainfall from NASA POWER climate data "
           "(free, no API key).")
l1, l2 = st.columns(2)
with l1:
    city = st.text_input("City", placeholder="e.g. Lucknow,IN", key="city")
with l2:
    season_name = st.selectbox("Growing season", list(SEASONS) + ["Custom months"], key="season")

if season_name == "Custom months":
    picked = st.multiselect("Months", MONTH_NAMES, default=["Jun", "Jul", "Aug"],
                            key="custom_months")
    months = [MONTH_NAMES.index(m) + 1 for m in picked]
else:
    months = SEASONS[season_name]

if st.button("Get seasonal climate", key="fetch_climate"):
    st.session_state["climate_result"] = None
    st.session_state["climate_error"] = None
    try:
        result = climate_for_city(city, months)
        result["notes"] = apply_to_sliders({
            "temperature": result["temperature"],
            "humidity": result["humidity"],
            "rainfall": result["rainfall"],
        })
        st.session_state["climate_result"] = result
    except ClimateError as err:
        st.session_state["climate_error"] = str(err)

if st.session_state["climate_error"]:
    st.error(st.session_state["climate_error"])

res = st.session_state["climate_result"]
if res:
    st.success(
        f"{res['place']}: season average {res['temperature']:.1f} °C, "
        f"{res['humidity']:.0f}% humidity, {res['rainfall']:.0f} mm rain per month. "
        "Filled in below."
    )
    for note in res["notes"]:
        st.info(note)
    with st.expander("Month-by-month climate"):
        st.dataframe(pd.DataFrame(res["monthly"]), hide_index=True)

# ---------------------------------------------------------------- optional live weather
with st.expander("Optional: today's weather from OpenWeatherMap (needs an API key)"):
    o1, o2 = st.columns([3, 1])
    with o1:
        owm_city = st.text_input("City for current weather", placeholder="e.g. Lucknow,IN",
                                 key="owm_city", label_visibility="collapsed")
    with o2:
        owm_fetch = st.button("Fetch weather", key="fetch_owm")
    st.caption("Fills today's temperature and humidity only. The seasonal data above "
               "is a better basis for choosing a crop.")
    if owm_fetch:
        try:
            w = get_current_weather(owm_city, api_key=get_weather_key())
            apply_to_sliders({"temperature": w["temperature"], "humidity": w["humidity"]})
            st.success(f"{w['city']}, {w['country']}: {w['temperature']:.1f} °C, "
                       f"{w['humidity']:.0f}% humidity, {w['description']}. "
                       "Temperature and humidity updated below.")
        except WeatherError as err:
            st.error(str(err))

# ---------------------------------------------------------------- field details
st.subheader("Enter your field details")
st.caption("Soil values (N, P, K, pH) can't come from a weather service. "
           "Use a soil test, for example the Soil Health Card in India.")
c1, c2 = st.columns(2)
with c1:
    N = st.number_input("Nitrogen (N)", 0, 200, 90)
    P = st.number_input("Phosphorus (P)", 0, 200, 42)
    K = st.number_input("Potassium (K)", 0, 250, 43)
    ph = st.slider("Soil pH", 3.5, 9.5, 6.5, 0.1)
with c2:
    temperature = st.slider("Temperature (°C)", 5.0, 45.0, step=0.5, key="temperature")
    humidity = st.slider("Humidity (%)", 10.0, 100.0, step=1.0, key="humidity")
    rainfall = st.slider(
        "Rainfall (mm per month)", 15.0, 300.0, step=1.0, key="rainfall",
        help="Treated as average monthly rainfall during the growing season.",
    )

if st.button("Get recommendation", type="primary"):
    inputs = dict(N=N, P=P, K=K, temperature=temperature,
                  humidity=humidity, ph=ph, rainfall=rainfall)
    results = recommend(inputs, top_k=3, model=model, features=features)

    best_crop, best_p = results[0]
    st.success(f"Recommended crop: **{best_crop.title()}** ({best_p:.0%} confidence)")

    chart_df = pd.DataFrame(
        {"Crop": [c.title() for c, _ in results], "Confidence": [p for _, p in results]}
    ).set_index("Crop")
    st.bar_chart(chart_df)

    if best_p < 0.6:
        st.warning("Low confidence: these conditions sit between crops. "
                   "Consider a local soil test and advice from an agriculture officer.")

st.divider()
st.caption("Agrisense is a decision aid, not a substitute for local agronomy advice.")
