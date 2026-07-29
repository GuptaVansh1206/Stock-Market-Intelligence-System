"""
Generates SHAP explanations for each trained model to show which
features drive the up/down predictions.
"""

import os
import numpy as np
import pandas as pd
import shap

from src.config import SHAP_DIR


def generate_shap_explanations(results: dict) -> dict:
    shap_results = {}

    for ticker, output in results.items():
        print(f"Generating SHAP for {ticker}")

        model = output["pipeline"].named_steps["model"]
        X_test = output["X_test"]

        explainer = shap.Explainer(model)
        explanation = explainer(X_test)

        shap_results[ticker] = {
            "explainer": explainer,
            "explanation": explanation,
            "X_test": X_test,
        }

    print("SHAP completed successfully.")
    return shap_results


def compute_shap_importance(shap_results: dict) -> dict:
    shap_importance = {}

    for ticker, data in shap_results.items():
        explanation = data["explanation"]
        X_test = data["X_test"]

        importance = np.abs(explanation.values).mean(axis=0)
        if importance.ndim == 2:
            importance = importance.mean(axis=1)

        importance_df = pd.DataFrame({
            "Feature": X_test.columns,
            "Importance": importance,
        }).sort_values("Importance", ascending=False)

        shap_importance[ticker] = importance_df

    return shap_importance


def save_shap_importance(shap_importance: dict):
    os.makedirs(SHAP_DIR, exist_ok=True)

    for ticker, df in shap_importance.items():
        filename = f"{SHAP_DIR}/{ticker.replace('^', '').replace('.', '_')}_importance.csv"
        df.to_csv(filename, index=False)

    print("SHAP feature importance saved.")
