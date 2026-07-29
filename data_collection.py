"""
Downloads 5 years of historical stock data (yfinance) and recent news
articles (NewsAPI) for each tracked asset, saving both to data/.
"""

import os
import requests
import pandas as pd
import yfinance as yf

from src.config import ASSETS, NEWS_API_KEY, STOCK_DATA_DIR, NEWS_DATA_DIR


def create_project_dirs():
    os.makedirs(STOCK_DATA_DIR, exist_ok=True)
    os.makedirs(NEWS_DATA_DIR, exist_ok=True)
    print("Project folders created successfully!")


def download_stock_data(assets=ASSETS, period="5y"):
    """Downloads historical OHLCV data for each ticker and saves as CSV."""
    stock_data = {}

    for ticker, company in assets.items():
        print(f"Downloading {company} ({ticker})...")

        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        df.reset_index(inplace=True)

        file_name = f"{STOCK_DATA_DIR}/{ticker.replace('^', '').replace('.', '_')}.csv"
        df.to_csv(file_name, index=False)

        stock_data[ticker] = df
        print(f"Saved to {file_name}")

    print("\nAll datasets downloaded successfully!")
    return stock_data


def fetch_news(company, api_key, page_size=30):
    """Fetches recent news articles for a company via NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": api_key,
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print(f"Error fetching {company}: {response.status_code}")
        return pd.DataFrame()

    articles = response.json().get("articles", [])

    news_list = [
        {
            "Company": company,
            "Source": article["source"]["name"],
            "Author": article.get("author"),
            "Title": article.get("title"),
            "Description": article.get("description"),
            "Published At": article.get("publishedAt"),
            "URL": article.get("url"),
        }
        for article in articles
    ]

    return pd.DataFrame(news_list)


def download_all_news(assets=ASSETS, api_key=NEWS_API_KEY):
    """Fetches and saves news for every tracked company, plus a combined file."""
    if not api_key:
        raise ValueError(
            "NEWS_API_KEY is not set. Copy .env.example to .env and add your key."
        )

    all_news = []

    for ticker, company in assets.items():
        print(f"Fetching news for {company}...")
        news_df = fetch_news(company, api_key)

        if len(news_df) > 0:
            file_name = f"{NEWS_DATA_DIR}/{company.replace(' ', '_')}_news.csv"
            news_df.to_csv(file_name, index=False)
            all_news.append(news_df)
            print(f"Saved {len(news_df)} articles")
        else:
            print("No news found.")

    combined_news = pd.concat(all_news, ignore_index=True)
    combined_news.to_csv(f"{NEWS_DATA_DIR}/all_news.csv", index=False)

    print(f"Total Articles: {len(combined_news)}")
    return combined_news


if __name__ == "__main__":
    create_project_dirs()
    download_stock_data()
    download_all_news()
