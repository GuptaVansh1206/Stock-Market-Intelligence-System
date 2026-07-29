"""
Builds price, moving-average, volatility, time, and volume features,
plus the binary next-day-direction target used for modeling.
"""

import numpy as np

from src.config import STOCK_DATA_DIR


def engineer_features(clean_stock_data: dict) -> dict:
    featured_data = {}

    for ticker, df in clean_stock_data.items():
        df = df.copy()

        # Price features
        df["Daily Return"] = df["Close"].pct_change()
        df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df["High-Low Ratio"] = df["High"] / df["Low"]

        # Moving averages
        df["MA_7"] = df["Close"].rolling(7).mean()
        df["MA_14"] = df["Close"].rolling(14).mean()
        df["MA_30"] = df["Close"].rolling(30).mean()
        df["Price_to_MA7"] = df["Close"] / df["MA_7"]
        df["MA_Crossover"] = (df["MA_7"] > df["MA_30"]).astype(int)

        # Volatility
        df["Volatility_7"] = df["Daily Return"].rolling(7).std()
        df["Volatility_14"] = df["Daily Return"].rolling(14).std()

        # Time features
        df["Day_of_Week"] = df["Date"].dt.dayofweek
        df["Month"] = df["Date"].dt.month
        df["Quarter"] = df["Date"].dt.quarter
        df["Is_Monday"] = (df["Day_of_Week"] == 0).astype(int)
        df["Is_Friday"] = (df["Day_of_Week"] == 4).astype(int)

        # Volume features
        df["Volume_MA7"] = df["Volume"].rolling(7).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA7"]

        # Target: did the price go up the next day?
        df["Tomorrow_Close"] = df["Close"].shift(-1)
        df["Target"] = (df["Tomorrow_Close"] > df["Close"]).astype(int)

        df = df.dropna().reset_index(drop=True)
        featured_data[ticker] = df

    print("Feature Engineering Completed!")
    return featured_data


def save_featured_data(featured_data: dict):
    for ticker, df in featured_data.items():
        filename = f"{STOCK_DATA_DIR}/featured_{ticker.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(filename, index=False)
    print("Feature-engineered datasets saved successfully.")
