"""
Finds the classification threshold per asset that minimizes total
misclassification cost (false positives + false negatives).
"""

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix

from src.config import MODELS_DIR, ASSETS


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.10, 0.91, 0.01)
    threshold_results = []

    for threshold in thresholds:
        predictions = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()

        cost = fp + fn
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        threshold_results.append({"Threshold": threshold, "Accuracy": accuracy, "Cost": cost})

    results_df = pd.DataFrame(threshold_results)
    best_row = results_df.loc[results_df["Cost"].idxmin()]

    return results_df, best_row


def tune_all_thresholds(results: dict) -> pd.DataFrame:
    best_thresholds = []

    for ticker, output in results.items():
        _, best = find_best_threshold(output["y_test"], output["y_proba"])

        best_thresholds.append({
            "Company": ASSETS.get(ticker, ticker),
            "Best Threshold": round(best["Threshold"], 2),
            "Accuracy": round(best["Accuracy"], 4),
            "Minimum Cost": int(best["Cost"]),
        })

    best_threshold_df = pd.DataFrame(best_thresholds)
    best_threshold_df.to_csv(f"{MODELS_DIR}/best_thresholds.csv", index=False)
    print("Best thresholds saved successfully.")

    return best_threshold_df
