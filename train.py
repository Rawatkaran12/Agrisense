"""Train the Agrisense crop recommendation model.

Usage:  python src/train.py
Reads   data/crop_data.csv
Writes  models/crop_model.joblib, models/metrics.json,
        models/confusion_matrix.png, models/feature_importance.png
"""
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "crop_data.csv"
MODELS = ROOT / "models"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def main():
    if not DATA.exists():
        raise SystemExit(f"{DATA} not found. Run: python src/generate_data.py")

    df = pd.read_csv(DATA)
    X, y = df[FEATURES], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Baseline vs. final model
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_train, y_train)
    base_acc = accuracy_score(y_test, baseline.predict(X_test))

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    cv = cross_val_score(model, X_train, y_train, cv=5)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "baseline_logreg_accuracy": round(base_acc, 4),
        "random_forest_cv_accuracy": round(float(cv.mean()), 4),
        "random_forest_test_accuracy": round(accuracy_score(y_test, pred), 4),
        "random_forest_test_f1_macro": round(f1_score(y_test, pred, average="macro"), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    MODELS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, MODELS / "crop_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Agrisense - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(MODELS / "confusion_matrix.png", dpi=120)
    plt.close()

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(6, 4))
    imp.plot(kind="barh", color="#2e7d32")
    plt.title("Agrisense - Feature Importance")
    plt.tight_layout()
    plt.savefig(MODELS / "feature_importance.png", dpi=120)
    plt.close()

    print(json.dumps(metrics, indent=2))
    print("\n", classification_report(y_test, pred))


if __name__ == "__main__":
    main()
