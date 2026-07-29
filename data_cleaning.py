"""
Cleans raw stock and news data: type conversion, dedup, missing values.
"""

import pandas as pd

from src.config import STOCK_DATA_DIR, NEWS_DATA_DIR


def clean_stock_data(stock_data: dict) -> dict:
    """Flattens columns, fixes dtypes, dedups, sorts, and forward-fills gaps."""
    clean_data = {}

    for ticker, df in stock_data.items():
        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates().sort_values("Date").reset_index(drop=True)

        df = df.ffill().dropna()

        clean_data[ticker] = df

    print("Stock datasets cleaned successfully.")
    return clean_data


def save_clean_stock_data(clean_data: dict):
    for ticker, df in clean_data.items():
        filename = f"{STOCK_DATA_DIR}/clean_{ticker.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(filename, index=False)
    print("Clean stock CSV files saved.")


def clean_news_data(combined_news: pd.DataFrame) -> pd.DataFrame:
    """Dedupes articles, drops rows missing title/description, fills missing authors."""
    news_clean = combined_news.copy()

    news_clean["Published At"] = pd.to_datetime(news_clean["Published At"])
    news_clean = news_clean.drop_duplicates()
    news_clean = news_clean.dropna(subset=["Title", "Description"])
    news_clean["Author"] = news_clean["Author"].fillna("Unknown")
    news_clean = news_clean.reset_index(drop=True)

    print("News dataset cleaned successfully.")
    return news_clean


def save_clean_news_data(news_clean: pd.DataFrame):
    news_clean.to_csv(f"{NEWS_DATA_DIR}/clean_all_news.csv", index=False)
    print("Clean news dataset saved successfully.")
