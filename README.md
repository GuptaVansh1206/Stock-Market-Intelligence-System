# Stock Market Prediction + News RAG QA System

An end-to-end project that:

1. **Collects** 5 years of historical price data (via `yfinance`) and recent news articles (via `NewsAPI`) for five Indian market assets (Sensex, Reliance, TCS, Infosys, HDFC Bank).
2. **Cleans & explores** the data (missing values, distributions, correlations, volume/price trends).
3. **Engineers features** — moving averages, volatility, return-based and calendar features — and a next-day up/down direction target.
4. **Trains** a `RandomForestClassifier` per asset (in a scaled `sklearn` pipeline) to predict next-day price direction, with a time-based train/test split.
5. **Tunes classification thresholds** per asset to minimize misclassification cost.
6. **Explains predictions** with SHAP (feature importance, beeswarm, waterfall plots).
7. **Answers natural-language questions about the news** using a Retrieval-Augmented Generation (RAG) pipeline: articles are embedded with `sentence-transformers`, indexed with `FAISS`, and questions are answered by Cohere's chat API grounded only in retrieved context.

## Project structure

```
stock-market-rag-analysis/
├── app.py                      # Streamlit app (UI on top of the pipeline)
├── main.py                     # Runs the full pipeline end to end (CLI)
├── requirements.txt
├── .env.example                # Template for your API keys (local runs)
├── .streamlit/
│   ├── config.toml             # Theme
│   └── secrets.toml.example    # Template for Streamlit Cloud secrets
├── src/
│   ├── config.py               # Assets, paths, feature list, env-based API keys
│   ├── data_collection.py      # yfinance + NewsAPI downloads
│   ├── data_cleaning.py        # Cleaning for stock & news data
│   ├── feature_engineering.py  # Technical indicators + target variable
│   ├── train_model.py          # RandomForest training pipeline per asset
│   ├── threshold_tuning.py     # Cost-minimizing threshold search
│   ├── explainability.py       # SHAP explanations
│   ├── news_database.py        # Richer news dataset for the RAG pipeline
│   └── rag_qa.py                # FAISS + Cohere RAG Q&A system
├── notebooks/
│   └── capstone_project.ipynb  # Original exploratory notebook
├── data/                        # Downloaded/cleaned CSVs (gitignored)
├── models/                      # Trained models + metrics (gitignored)
└── shap_outputs/                # SHAP importance CSVs (gitignored)
```

## Setup

```bash
git clone https://github.com/<your-username>/stock-market-rag-analysis.git
cd stock-market-rag-analysis

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env       # then fill in your own API keys
```

You'll need free API keys from:
- [NewsAPI](https://newsapi.org/) → `NEWS_API_KEY`
- [Cohere](https://cohere.com/) → `COHERE_API_KEY`

**Never commit your `.env` file** — it's already gitignored.

## Usage

Run the full pipeline:

```bash
python main.py
```

Or explore step by step in `notebooks/capstone_project.ipynb`.

Use the RAG system directly in Python:

```python
from src.news_database import build_news_database
from src.rag_qa import NewsRAG

news_database = build_news_database()
rag = NewsRAG(news_database)

print(rag.answer("Why is Reliance in the news?"))
```

## Streamlit app

`app.py` puts a UI on top of the same pipeline: price/volume charts, next-day
prediction with confidence, SHAP explainability, and a news Q&A tab.

**Run locally:**

```bash
pip install -r requirements.txt
cp .env.example .env       # fill in your keys
streamlit run app.py
```

Then click **Run / refresh full pipeline** in the sidebar (first run takes a
few minutes — it downloads data, trains models, and computes SHAP values;
results are cached after that within the session).

**Deploy on Streamlit Community Cloud:**

1. Push this repo to GitHub (see below).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app."
3. Pick this repo/branch and set the main file path to `app.py`.
4. Before or after the first deploy, open **Settings → Secrets** on the app
   and paste in:
   ```toml
   NEWS_API_KEY = "your_newsapi_key_here"
   COHERE_API_KEY = "your_cohere_key_here"
   ```
   (same format as `.streamlit/secrets.toml.example`). `app.py` reads these
   from `st.secrets` and wires them into the environment automatically — you
   don't need a `.env` file on Cloud.
5. Deploy. Note that yfinance/NewsAPI calls and model training happen live in
   the app process, so the first run after each deploy will be slow; consider
   pre-generating `data/`, `models/`, and `shap_outputs/` locally and
   committing them (remove the relevant lines from `.gitignore`) if you want
   the deployed app to start pre-loaded instead of training on first click.

## Notes

- Model predictions (next-day up/down direction) are for educational purposes only and are **not financial advice**.
- A security note: an earlier version of this project's notebook had API keys hardcoded directly in the code. Those have been removed and replaced with environment variables — if you're reusing keys from an old copy of the notebook, regenerate them.

## License

MIT — see [LICENSE](LICENSE).
