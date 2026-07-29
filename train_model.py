"""
Trains a RandomForest classifier (in a scaling pipeline) per asset to
predict next-day price direction, using a time-based train/test split.
"""

import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.config import FEATURE_COLUMNS, MODELS_DIR, ASSETS


def process_asset(featured_data: dict, ticker: str) -> dict:
    df = featured_data[ticker].copy()

    X = df[FEATURE_COLUMNS]
    y = df["Target"]

    # Time-based split (no shuffling — this is a time series)
    split_index = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, random_state=42, max_depth=10)),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_proba),
    }

    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_all_assets(featured_data: dict) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    results = {}

    for ticker in featured_data.keys():
        print(f"Training Model : {ASSETS.get(ticker, ticker)}")
        output = process_asset(featured_data, ticker)
        results[ticker] = output

        for name, value in output["metrics"].items():
            print(f"  {name:10}: {value:.4f}")

    return results


def save_models_and_metrics(results: dict):
    comparison = []

    for ticker, output in results.items():
        filename = f"{MODELS_DIR}/{ticker.replace('^', '').replace('.', '_')}_model.pkl"
        joblib.dump(output["pipeline"], filename)

        comparison.append({
            "Company": ASSETS.get(ticker, ticker),
            "Accuracy": output["metrics"]["Accuracy"],
            "Precision": output["metrics"]["Precision"],
            "Recall": output["metrics"]["Recall"],
            "F1 Score": output["metrics"]["F1"],
            "ROC AUC": output["metrics"]["ROC_AUC"],
        })

    print("All models saved successfully.")

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(f"{MODELS_DIR}/model_metrics.csv", index=False)
    print("Metrics saved successfully.")

    return comparison_df
