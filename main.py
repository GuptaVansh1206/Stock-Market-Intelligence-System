"""
Runs the full pipeline end to end:
  1. Collect stock + news data
  2. Clean it
  3. Engineer features
  4. Train per-asset models
  5. Tune classification thresholds
  6. Generate SHAP explanations
  7. Build the RAG news Q&A system and ask a sample question

Requires a .env file with NEWS_API_KEY and COHERE_API_KEY set
(see .env.example).
"""

from src.data_collection import create_project_dirs, download_stock_data, download_all_news
from src.data_cleaning import clean_stock_data, save_clean_stock_data, clean_news_data, save_clean_news_data
from src.feature_engineering import engineer_features, save_featured_data
from src.train_model import train_all_assets, save_models_and_metrics
from src.threshold_tuning import tune_all_thresholds
from src.explainability import generate_shap_explanations, compute_shap_importance, save_shap_importance
from src.news_database import build_news_database
from src.rag_qa import NewsRAG


def main():
    # 1. Collect data
    create_project_dirs()
    stock_data = download_stock_data()
    combined_news = download_all_news()

    # 2. Clean data
    clean_stock = clean_stock_data(stock_data)
    save_clean_stock_data(clean_stock)

    clean_news = clean_news_data(combined_news)
    save_clean_news_data(clean_news)

    # 3. Feature engineering
    featured_data = engineer_features(clean_stock)
    save_featured_data(featured_data)

    # 4. Train models
    results = train_all_assets(featured_data)
    save_models_and_metrics(results)

    # 5. Threshold tuning
    tune_all_thresholds(results)

    # 6. Explainability
    shap_results = generate_shap_explanations(results)
    shap_importance = compute_shap_importance(shap_results)
    save_shap_importance(shap_importance)

    # 7. RAG news Q&A
    news_database = build_news_database()
    rag = NewsRAG(news_database)

    question = "Why is Reliance in the news?"
    print(f"\nQ: {question}")
    print("A:", rag.answer(question))


if __name__ == "__main__":
    main()
