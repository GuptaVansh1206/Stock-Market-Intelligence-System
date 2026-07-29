# 📈 Stock Market Intelligence System

An end-to-end AI-powered Stock Market Intelligence System that predicts stock market direction using Machine Learning, explains predictions using SHAP Explainability, answers financial questions using Retrieval-Augmented Generation (RAG), compares multiple stocks, and provides an interactive Streamlit web application.

---

# 🚀 Project Overview

This project was developed as a Capstone Project to demonstrate practical applications of:

* Machine Learning
* Financial Data Analysis
* Explainable AI (XAI)
* Large Language Models (LLMs)
* Retrieval-Augmented Generation (RAG)
* Streamlit Deployment

The application combines stock market prediction with modern AI technologies to provide an intelligent financial analysis platform.

---

# 🎯 Problem Statement

Build an intelligent system capable of:

* Predicting stock market movement (UP/DOWN)
* Explaining predictions using SHAP
* Answering stock-related questions using RAG + Cohere
* Comparing multiple assets
* Displaying everything through an interactive Streamlit application

---

# 📊 Assets Used

* Sensex (^BSESN)
* Reliance Industries
* Tata Consultancy Services (TCS)
* Infosys
* HDFC Bank

---

# 📰 Data Sources

### Stock Data

* Yahoo Finance (yfinance)

### News Data

* NewsAPI

---

# ⚙️ Technologies Used

## Programming Language

* Python 3.11+

## Libraries

* pandas
* numpy
* yfinance
* requests
* scikit-learn
* matplotlib
* plotly
* shap
* faiss-cpu
* sentence-transformers
* cohere
* streamlit
* joblib
* nltk

---

# 🧠 Machine Learning Pipeline

The project follows a complete ML workflow.

## 1. Data Collection

* Historical stock prices
* Financial news headlines

## 2. Data Cleaning

* Missing value handling
* Date formatting
* Feature selection

## 3. Feature Engineering

Generated features include:

* Daily Return
* Log Return
* Moving Average (7, 14, 30)
* Volatility
* RSI
* MACD
* Bollinger Bands
* Volume Ratio
* High-Low Ratio
* Day of Week
* Month
* Quarter

## 4. Model Building

Random Forest Classifier

Pipeline:

```
Features
      ↓
StandardScaler
      ↓
Random Forest
      ↓
Prediction
```

---

# 📈 Business Metrics

The project evaluates models using

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score

Threshold tuning is also performed to minimize business cost.

---

# 🔍 Explainable AI

The model predictions are explained using SHAP.

Visualizations include:

* Feature Importance
* SHAP Summary Plot
* Waterfall Plot
* Force Plot

---

# 🤖 Retrieval-Augmented Generation (RAG)

The RAG system uses:

* Sentence Transformers
* FAISS Vector Database
* Cohere Command Model

Workflow:

```
User Question

        ↓

Sentence Embedding

        ↓

FAISS Similarity Search

        ↓

Top Relevant News

        ↓

Cohere LLM

        ↓

Final Answer
```

---

# 💻 Streamlit Application

The application contains multiple sections including:

### 📊 Predictions

* Stock Selection
* Price Trend
* Prediction
* Confidence Score
* SHAP Explanation

### 💬 AI Chat

Ask questions like:

* Why is Reliance predicted to go UP?
* What happened in March 2024?
* Compare TCS and Infosys.
* Summarize today's financial news.

### 📈 Stock Comparison

Compare

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC

---

# 📂 Project Structure

```
Stock-Market-Intelligence-System/

│── app.py
│── config.py
│── requirements.txt
│── README.md
│── notebooks/
│── src/
│── models/
│── data/
│── outputs/
│── screenshots/
│── assets/
│── docs/
```

---

# ▶️ Installation

Clone the repository

```
git clone https://github.com/your-username/Stock-Market-Intelligence-System.git
```

Move into the project folder

```
cd Stock-Market-Intelligence-System
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app.py
```

---

# 📷 Screenshots

Add screenshots of

* Home Page
* Prediction Dashboard
* SHAP Visualizations
* Chat Interface
* Comparison Dashboard

inside the `screenshots/` folder.

---

# 📌 Future Improvements

* LSTM-based prediction
* Transformer-based forecasting
* Live market streaming
* Portfolio optimization
* Real-time sentiment analysis
* Cloud deployment
* Multi-language chatbot
* Email alerts

---

# 👨‍💻 Author

**Vansh Gupta**

Computer Science (Data Science)

---

# 📄 License

This project is licensed under the MIT License.

---

# ⭐ Acknowledgements

* Yahoo Finance
* NewsAPI
* Cohere
* FAISS
* Sentence Transformers
* Streamlit
* SHAP
* Scikit-Learn

---

If you find this project useful, consider giving the repository a ⭐ on GitHub.
