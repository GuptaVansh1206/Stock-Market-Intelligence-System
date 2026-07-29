"""
Builds the richer news dataset (with full article Content, not just
Description) used specifically by the RAG pipeline in rag_qa.py.
"""

import os
import pandas as pd

from newsapi import NewsApiClient

from src.config import NEWS_API_KEY, STOCK_QUERIES, NEWS_DATA_DIR


def build_news_database(stock_queries: dict = STOCK_QUERIES, page_size: int = 100) -> dict:
    if not NEWS_API_KEY:
        raise ValueError(
            "NEWS_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    os.makedirs(NEWS_DATA_DIR, exist_ok=True)
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    news_database = {}

    for ticker, query in stock_queries.items():
        print(f"Downloading news for {query}")

        articles = newsapi.get_everything(
            q=query, language="en", sort_by="publishedAt", page_size=page_size
        )

        rows = [
            {
                "Ticker": ticker,
                "Company": query,
                "Title": article["title"],
                "Description": article["description"],
                "Content": article["content"],
                "Source": article["source"]["name"],
                "Date": article["publishedAt"],
            }
            for article in articles["articles"]
        ]

        df = pd.DataFrame(rows)
        news_database[ticker] = df

        filename = f"{NEWS_DATA_DIR}/{ticker.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(filename, index=False)

    print("News Download Completed")
    return news_database
