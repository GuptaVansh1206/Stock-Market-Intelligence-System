"""
Central configuration for the project.

API keys are loaded from environment variables (never hardcoded).
Copy .env.example to .env and fill in your own keys before running anything
that hits NewsAPI or Cohere.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ----------------------------------------------------------------
# API Keys (set these in a local .env file, which is gitignored)
# ----------------------------------------------------------------
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# ----------------------------------------------------------------
# Assets tracked by the project
# ----------------------------------------------------------------
ASSETS = {
    "^BSESN": "Sensex",
    "RELIANCE.NS": "Reliance",
    "TCS.NS": "TCS",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
}

# Descriptive search queries used for the news / RAG pipeline
STOCK_QUERIES = {
    "^BSESN": "BSE Sensex",
    "RELIANCE.NS": "Reliance Industries",
    "TCS.NS": "Tata Consultancy Services",
    "INFY.NS": "Infosys",
    "HDFCBANK.NS": "HDFC Bank",
}

# ----------------------------------------------------------------
# Paths
# ----------------------------------------------------------------
DATA_DIR = "data"
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_data")
NEWS_DATA_DIR = os.path.join(DATA_DIR, "news_data")
MODELS_DIR = "models"
SHAP_DIR = "shap_outputs"

# ----------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------
FEATURE_COLUMNS = [
    "Daily Return",
    "Log Return",
    "High-Low Ratio",
    "MA_7",
    "MA_14",
    "MA_30",
    "Price_to_MA7",
    "MA_Crossover",
    "Volatility_7",
    "Volatility_14",
    "Day_of_Week",
    "Month",
    "Quarter",
    "Is_Monday",
    "Is_Friday",
    "Volume_MA7",
    "Volume_Ratio",
]

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
COHERE_CHAT_MODEL = "command-a-03-2025"
