"""Generate a SYNTHETIC crop dataset so the project runs out of the box.

The values are approximate typical ranges per crop, modelled on the public
Kaggle "Crop Recommendation Dataset". For a real project, download that
dataset and save it as data/crop_data.csv (same column names) instead.

Columns: N, P, K, temperature, humidity, ph, rainfall, label
"""
from pathlib import Path

import numpy as np
import pandas as pd

# crop: (N, P, K, temperature C, humidity %, pH, rainfall mm)
CROP_PROFILES = {
    "rice": (80, 48, 40, 23.7, 82, 6.4, 236),
    "maize": (78, 48, 20, 22.4, 65, 6.2, 85),
    "chickpea": (40, 68, 80, 18.9, 17, 7.3, 80),
    "kidneybeans": (21, 68, 20, 20.0, 22, 5.7, 106),
    "pigeonpeas": (21, 68, 20, 27.7, 49, 5.8, 149),
    "mothbeans": (21, 48, 20, 28.2, 53, 6.8, 51),
    "mungbean": (21, 48, 20, 28.5, 85, 6.7, 48),
    "blackgram": (40, 68, 19, 30.0, 65, 7.1, 68),
    "lentil": (19, 68, 19, 24.5, 64, 6.9, 46),
    "pomegranate": (19, 19, 40, 21.8, 90, 6.4, 108),
    "banana": (100, 82, 50, 27.4, 80, 6.0, 105),
    "mango": (20, 28, 30, 31.2, 50, 5.8, 95),
    "grapes": (23, 133, 200, 23.8, 82, 6.0, 70),
    "watermelon": (99, 17, 50, 25.6, 85, 6.5, 50),
    "muskmelon": (100, 17, 50, 28.7, 92, 6.4, 25),
    "apple": (21, 134, 200, 22.6, 92, 5.9, 113),
    "orange": (20, 16, 10, 22.8, 92, 7.0, 110),
    "papaya": (50, 59, 50, 33.7, 92, 6.7, 143),
    "coconut": (22, 17, 31, 27.4, 95, 6.0, 175),
    "cotton": (118, 46, 20, 24.0, 80, 6.9, 80),
    "jute": (78, 46, 40, 25.0, 80, 6.7, 175),
    "coffee": (101, 29, 30, 25.5, 58, 6.8, 158),
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def generate(samples_per_crop: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for crop, (n, p, k, temp, hum, ph, rain) in CROP_PROFILES.items():
        data = {
            "N": rng.normal(n, max(4, n * 0.12), samples_per_crop),
            "P": rng.normal(p, max(4, p * 0.12), samples_per_crop),
            "K": rng.normal(k, max(4, k * 0.12), samples_per_crop),
            "temperature": rng.normal(temp, 2.0, samples_per_crop),
            "humidity": rng.normal(hum, 5.0, samples_per_crop),
            "ph": rng.normal(ph, 0.3, samples_per_crop),
            "rainfall": rng.normal(rain, max(5, rain * 0.15), samples_per_crop),
        }
        df = pd.DataFrame(data)
        df["label"] = crop
        rows.append(df)

    df = pd.concat(rows, ignore_index=True)
    df[["N", "P", "K"]] = df[["N", "P", "K"]].clip(0, 200).round(0)
    df["temperature"] = df["temperature"].clip(5, 45).round(2)
    df["humidity"] = df["humidity"].clip(10, 100).round(2)
    df["ph"] = df["ph"].clip(3.5, 9.5).round(2)
    df["rainfall"] = df["rainfall"].clip(15, 300).round(2)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "data" / "crop_data.csv"
    out.parent.mkdir(exist_ok=True)
    df = generate()
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows ({df['label'].nunique()} crops) to {out}")
