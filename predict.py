"""Load the trained model and recommend crops.

Usage (CLI):
    python src/predict.py --N 90 --P 42 --K 43 --temperature 24 --humidity 80 --ph 6.5 --rainfall 200
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "crop_model.joblib"


def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Model not found. Run: python src/train.py")
    bundle = joblib.load(path)
    return bundle["model"], bundle["features"]


def recommend(inputs: dict, top_k: int = 3, model=None, features=None):
    """Return the top_k crops as a list of (crop, probability)."""
    if model is None:
        model, features = load_model()
    row = pd.DataFrame([[inputs[f] for f in features]], columns=features)
    probs = model.predict_proba(row)[0]
    ranked = sorted(zip(model.classes_, probs), key=lambda t: t[1], reverse=True)
    return [(crop, float(p)) for crop, p in ranked[:top_k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrisense crop recommendation")
    for name in ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]:
        parser.add_argument(f"--{name}", type=float, required=True)
    args = vars(parser.parse_args())
    for i, (crop, p) in enumerate(recommend(args), 1):
        print(f"{i}. {crop.title():<12} {p:.1%}")
