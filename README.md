# 🌾 Agrisense

Crop recommendation from soil nutrients (N, P, K, pH) and climate (temperature, humidity,
rainfall). Enter a city and a growing season and Agrisense fills in the climate values from
NASA POWER data; you add the soil values and get the top 3 crops with confidence scores.

## How it works

```
City + season ──► Open-Meteo geocoding ──► NASA POWER climatology ──┐
                                                                     ├─► Random Forest ──► top 3 crops
Soil test values (N, P, K, pH) ──────────────────────────────────────┘
```

| Input | Source |
|---|---|
| Temperature, humidity, rainfall | NASA POWER long-term monthly averages for the chosen season (free, no key) |
| N, P, K, pH | You enter them (soil test, e.g. the Soil Health Card in India) |
| Today's weather (optional) | OpenWeatherMap, needs a free API key |

## Project structure

```
agrisense/
├── data/crop_data.csv        # Kaggle Crop Recommendation Dataset (2,200 rows, 22 crops)
├── src/
│   ├── train.py              # trains and evaluates the model
│   ├── predict.py            # loads the model, recommends crops
│   ├── climate.py            # city -> coordinates -> NASA POWER seasonal climate
│   ├── weather.py            # optional OpenWeatherMap current weather
│   └── generate_data.py      # synthetic data (only for running without the real CSV)
├── app/streamlit_app.py      # web UI
├── tests/test_agrisense.py   # offline tests (all network calls mocked)
├── models/                   # saved model, metrics, confusion matrix, feature importance
├── notebooks/                # your EDA and experiments
└── requirements.txt
```

## Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt

python src/train.py              # optional: a trained model is already in models/
streamlit run app/streamlit_app.py
```

In the app: enter a city (e.g. `Lucknow,IN`), pick a season, click **Get seasonal climate**,
enter your soil values, then click **Get recommendation**.

Command-line check of the model:

```bash
python src/predict.py --N 90 --P 42 --K 43 --temperature 21 --humidity 82 --ph 6.5 --rainfall 203
python src/climate.py "Lucknow,IN" kharif
```

## Optional: OpenWeatherMap (today's weather)

NASA POWER needs no key. For the optional "today's weather" box:

1. Create a free account at openweathermap.org and copy your key (new keys can take up to
   a couple of hours to activate).
2. Set it as an environment variable `OPENWEATHER_API_KEY`, or put
   `OPENWEATHER_API_KEY = "your_key"` in `.streamlit/secrets.toml`.
   Both are already in `.gitignore`. Never put the key in the code.

## Model results (real Kaggle data)

| Model | Accuracy |
|---|---|
| Logistic Regression baseline | 97.3% (test) |
| Random Forest, 5-fold cross-validation | 99.5% |
| Random Forest, held-out test set (440 rows) | 99.3% (macro F1 99.3%) |

The dataset is clean, balanced (100 rows per crop) and very well separated, so these numbers
are higher than real farms would give. Treat them as a measure of how well the model learned
this dataset, not of field performance.

## Assumptions and limitations (read before presenting)

- **Rainfall is treated as average monthly rainfall (mm).** The dataset does not document its
  period. Values of 20 to 300 mm fit monthly totals far better than seasonal totals, but this is
  an interpretation. NASA POWER gives mm/day, which `climate.py` converts to mm per month.
- **Climate is a long-term average, not a forecast.** It describes typical conditions for the
  place and season, which matches what the model was trained on.
- **NASA POWER is gridded satellite and model data** (roughly 50 km cells), less precise than a
  local weather station. A city and its farmland can differ.
- **Values outside the slider range** (for example very wet places) are clamped, and the app says
  so.
- **No soil, region, irrigation, price or season-specific variety data.** The model only sees 7
  numbers. Confirm with local soil tests and agriculture officers before real decisions.
- The NASA POWER and geocoding calls were tested with mocked responses; run
  `python src/climate.py "Lucknow,IN" kharif` once on your machine to confirm live access.

## Tests

```bash
pip install pytest
pytest -q
```

## Roadmap

- [ ] Add a notebook with exploratory analysis and model comparison (XGBoost, SVM)
- [ ] Regional data and a soil-data source (Soil Health Card)
- [ ] Hindi / local-language interface
- [ ] Yield and price prediction
- [ ] Plant disease detection from leaf photos
- [ ] Deploy on Streamlit Community Cloud or Hugging Face Spaces
