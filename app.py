"""
Streamlit front-end for the stock market prediction + news RAG project.

Local dev:
    cp .env.example .env   # fill in your keys
    streamlit run app.py

Streamlit Community Cloud:
    Set NEWS_API_KEY and COHERE_API_KEY in the app's "Secrets" panel
    (Settings -> Secrets) using the same TOML format as
    .streamlit/secrets.toml.example. This file pulls them into
    os.environ before any src/ module is imported, so src/config.py
    picks them up the same way it does locally via .env.
"""

import os
import streamlit as st

# ----------------------------------------------------------------
# Pull secrets from Streamlit Cloud (if present) into env vars
# BEFORE importing anything from src/, since src/config.py reads
# os.getenv(...) at import time.
# ----------------------------------------------------------------
for key in ("NEWS_API_KEY", "COHERE_API_KEY"):
    if key in st.secrets:
        os.environ[key] = st.secrets[key]

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import ASSETS, FEATURE_COLUMNS, STOCK_DATA_DIR, MODELS_DIR, SHAP_DIR
from src.data_collection import create_project_dirs, download_stock_data, download_all_news
from src.data_cleaning import clean_stock_data, save_clean_stock_data, clean_news_data, save_clean_news_data
from src.feature_engineering import engineer_features, save_featured_data
from src.train_model import train_all_assets, save_models_and_metrics
from src.threshold_tuning import tune_all_thresholds
from src.explainability import generate_shap_explanations, compute_shap_importance, save_shap_importance
from src.news_database import build_news_database
from src.rag_qa import NewsRAG

st.set_page_config(
    page_title="Stock Market Prediction + News RAG",
    page_icon="📈",
    layout="wide",
)

# ----------------------------------------------------------------
# Cached pipeline steps
# ----------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_or_download_stock_data():
    create_project_dirs()
    raw = download_stock_data()
    cleaned = clean_stock_data(raw)
    save_clean_stock_data(cleaned)
    featured = engineer_features(cleaned)
    save_featured_data(featured)
    return featured


@st.cache_resource(show_spinner=False)
def _train_models(_featured_data):
    results = train_all_assets(_featured_data)
    metrics_df = save_models_and_metrics(results)
    thresholds_df = tune_all_thresholds(results)
    return results, metrics_df, thresholds_df


@st.cache_resource(show_spinner=False)
def _run_shap(_results):
    shap_results = generate_shap_explanations(_results)
    shap_importance = compute_shap_importance(shap_results)
    save_shap_importance(shap_importance)
    return shap_importance


@st.cache_resource(show_spinner=False)
def _build_rag():
    news_database = build_news_database()
    return NewsRAG(news_database)


# ----------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------

st.sidebar.title("📈 Controls")
ticker = st.sidebar.selectbox(
    "Asset", options=list(ASSETS.keys()), format_func=lambda t: f"{ASSETS[t]} ({t})"
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "First run downloads 5 years of price history + news and trains models — "
    "this can take a few minutes. Results are cached after that."
)
run_pipeline = st.sidebar.button("Run / refresh full pipeline", type="primary")

st.title("Stock Market Prediction & News Intelligence")
st.caption(
    "Next-day direction prediction (RandomForest + SHAP) and a news Q&A "
    "assistant (FAISS + Cohere RAG) for five Indian market assets."
)

if "featured_data" not in st.session_state:
    st.session_state.featured_data = None
    st.session_state.results = None
    st.session_state.metrics_df = None
    st.session_state.thresholds_df = None
    st.session_state.shap_importance = None

if run_pipeline:
    with st.spinner("Downloading & cleaning data, engineering features..."):
        st.session_state.featured_data = _load_or_download_stock_data()

    with st.spinner("Training models & tuning thresholds..."):
        results, metrics_df, thresholds_df = _train_models(st.session_state.featured_data)
        st.session_state.results = results
        st.session_state.metrics_df = metrics_df
        st.session_state.thresholds_df = thresholds_df

    with st.spinner("Generating SHAP explanations..."):
        st.session_state.shap_importance = _run_shap(st.session_state.results)

    st.success("Pipeline complete!")

# ----------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------

tab_prices, tab_predict, tab_explain, tab_qa = st.tabs(
    ["Price Trends", "Prediction", "Explainability", "News Q&A"]
)

# --- Price Trends ---
with tab_prices:
    if st.session_state.featured_data is None:
        st.info("Click **Run / refresh full pipeline** in the sidebar to load data.")
    else:
        df = st.session_state.featured_data[ticker]

        fig = px.line(
            df, x="Date", y="Close",
            title=f"{ASSETS[ticker]} — Closing Price",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.5, height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_vol = px.area(
                df, x="Date", y="Volume",
                title=f"{ASSETS[ticker]} — Trading Volume",
                template="plotly_white",
            )
            fig_vol.update_layout(title_x=0.5, height=400)
            st.plotly_chart(fig_vol, use_container_width=True)
        with col2:
            fig_ret = px.histogram(
                df, x="Daily Return", nbins=60,
                title=f"{ASSETS[ticker]} — Daily Return Distribution",
                template="plotly_white", marginal="box",
            )
            fig_ret.update_layout(title_x=0.5, height=400)
            st.plotly_chart(fig_ret, use_container_width=True)

# --- Prediction ---
with tab_predict:
    if st.session_state.results is None:
        st.info("Click **Run / refresh full pipeline** in the sidebar to train models.")
    else:
        output = st.session_state.results[ticker]
        metrics = output["metrics"]

        cols = st.columns(5)
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name, f"{value:.3f}")

        latest_row = st.session_state.featured_data[ticker].iloc[[-1]]
        pipeline = output["pipeline"]
        proba_up = pipeline.predict_proba(latest_row[FEATURE_COLUMNS])[0][1]

        st.subheader("Latest prediction")
        st.metric(
            f"Probability {ASSETS[ticker]} closes UP next session",
            f"{proba_up:.1%}",
        )
        st.caption("Educational demo only — not financial advice.")

        comparison = pd.DataFrame({
            "Actual": output["y_test"].values,
            "Predicted": output["y_pred"],
        }).reset_index(drop=True)
        comparison["Observation"] = comparison.index

        fig = px.line(
            comparison, x="Observation", y=["Actual", "Predicted"],
            title=f"Actual vs Predicted — {ASSETS[ticker]} (test set)",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.5, height=450)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.thresholds_df is not None:
            st.subheader("Tuned decision thresholds (all assets)")
            st.dataframe(st.session_state.thresholds_df, use_container_width=True)

# --- Explainability ---
with tab_explain:
    if st.session_state.shap_importance is None:
        st.info("Click **Run / refresh full pipeline** in the sidebar to generate SHAP values.")
    else:
        importance_df = st.session_state.shap_importance[ticker]

        fig = px.bar(
            importance_df, x="Importance", y="Feature", orientation="h",
            title=f"SHAP Feature Importance — {ASSETS[ticker]}",
            template="plotly_white",
        )
        fig.update_layout(title_x=0.5, height=600)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(importance_df, use_container_width=True)

# --- News Q&A ---
with tab_qa:
    st.write(
        "Ask a question about recent news for the tracked companies. "
        "Answers are grounded only in retrieved news articles."
    )

    if not os.getenv("NEWS_API_KEY") or not os.getenv("COHERE_API_KEY"):
        st.warning(
            "NEWS_API_KEY and/or COHERE_API_KEY are not set. Add them to your "
            ".env file locally, or to this app's Secrets if deployed on "
            "Streamlit Community Cloud."
        )
    else:
        question = st.text_input(
            "Your question", placeholder="Why is Reliance in the news?"
        )
        if st.button("Ask") and question:
            with st.spinner("Retrieving news & generating answer..."):
                rag = _build_rag()
                answer = rag.answer(question)
            st.markdown(f"**Answer:** {answer}")
