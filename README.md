# 📈 Stock Price Prediction using LSTM and RNN

A deep learning project that predicts Apple (AAPL) stock prices using Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks. Built end-to-end from data acquisition to a live Streamlit web application.

---

## 🚀 Live Demo

[👉 Click here to open the app]([https://your-streamlit-app-link.streamlit.app](https://stock-price-cmcghvsukcm9cryvmhnhjj.streamlit.app/))

---

## 📌 Problem Statement

Stock price forecasting is a classic time series problem. This project explores whether deep learning models — specifically RNN and LSTM architectures — can learn temporal patterns from historical AAPL stock data to predict the next day's closing price.

---

## 🏗️ Project Architecture

```
Data (yfinance)
    ↓
EDA & Feature Engineering (RSI, MACD, Bollinger Bands)
    ↓
Preprocessing (MinMaxScaler, Sliding Window Sequences)
    ↓
Model Training (SimpleRNN → Stacked LSTM → Multivariate LSTM)
    ↓
Evaluation (RMSE, MAE, MAPE, Directional Accuracy)
    ↓
Streamlit Deployment
```

---

## 📊 Dataset

- **Source:** Yahoo Finance via `yfinance`
- **Ticker:** AAPL (Apple Inc.)
- **Period:** January 2015 – December 2024
- **Features:** Open, High, Low, Close, Volume + Technical Indicators

---

## 🔧 Feature Engineering

Raw OHLCV data was extended with the following technical indicators:

| Indicator | Description | Why Used |
|---|---|---|
| RSI-14 | Relative Strength Index (14-day) | Momentum — overbought/oversold signal |
| MACD | Moving Average Convergence Divergence | Trend shift detection |
| MACD Signal | 9-day EMA of MACD | Confirms MACD crossovers |
| BB High | Bollinger Band Upper | Volatility upper bound |
| BB Low | Bollinger Band Lower | Volatility lower bound |

> **Note:** EMA-20 and EMA-50 were dropped after correlation analysis showed ~1.0 correlation with Close price — adding redundant information without value.

---

## ⚙️ Preprocessing

- **Scaler:** MinMaxScaler [0, 1] — chosen over StandardScaler because LSTM sigmoid/tanh activations align with [0,1] range
- **Split:** Chronological 70% / 15% / 15% train/val/test — never shuffled to preserve temporal order
- **Scaler fit:** Only on training data to prevent data leakage
- **Sequence length:** 60 trading days (lookback window)
- **Input shape:** `(samples, 60, 10)` — 60 timesteps, 10 features

---

## 🧠 Models

### Model 1 — Baseline SimpleRNN
```
SimpleRNN(64) → Dropout(0.2) → Dense(1)
```

### Model 2 — Stacked LSTM
```
LSTM(64) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(1)
```

### Model 3 — Multivariate LSTM
```
LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1)
```

**Training config:**
- Optimizer: Adam
- Loss: MSE
- Epochs: 100 (with EarlyStopping)
- Batch size: 32
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)

---

## 📈 Results

| Model | RMSE | MAE | MAPE | Directional Accuracy |
|---|---|---|---|---|
| SimpleRNN | $15.02 | $13.05 | 5.51% | 53.60% |
| Stacked LSTM | $48.97 | $46.04 | 19.44% | 49.57% |
| Multivariate LSTM | $44.64 | $41.26 | 17.33% | 50.43% |

**Streamlit App (Model 3):**
- RMSE: $11.67
- MAE: $7.19
- MAPE: 7.71%
- Directional Accuracy: 52.58%

> **Key Finding:** Contrary to expectation, SimpleRNN outperformed stacked LSTM models on this dataset. This is consistent with Occam's Razor in ML — simpler architectures generalize better when training data is limited relative to model complexity.

---

## 🎯 Honest Assessment

This project demonstrates time series forecasting with deep learning. Like all stock prediction models, it is subject to the **Efficient Market Hypothesis** — stock prices reflect all publicly available information, making consistent prediction extremely difficult.

- The model predicts next-day closing price based on the last 60 days of data
- A Directional Accuracy of ~53% is slightly better than random (50%) but not sufficient for real trading
- This is a **deep learning portfolio project**, not a trading system

---

## 🗂️ Project Structure

```
stock-price-prediction/
│
├── notebook.ipynb          # Full pipeline — EDA, preprocessing, training, evaluation
├── app.py                  # Streamlit web application
├── lstm_model.keras        # Saved trained model
├── scaler.pkl              # Saved MinMaxScaler
├── requirements.txt        # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python 3.11
- **Deep Learning:** TensorFlow, Keras
- **Data:** yfinance, pandas, numpy
- **Technical Indicators:** ta (Technical Analysis library)
- **Visualization:** matplotlib, seaborn
- **ML Utilities:** scikit-learn
- **Deployment:** Streamlit

---

## ⚡ Quick Start

**1. Clone the repo:**
```bash
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
```

**2. Create environment:**
```bash
conda create -n stock_env python=3.11 -y
conda activate stock_env
pip install -r requirements.txt
```

**3. Run the notebook:**
```bash
jupyter notebook notebook.ipynb
```

**4. Run the Streamlit app:**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
tensorflow
keras
pandas
numpy
matplotlib
seaborn
yfinance
ta
scikit-learn
joblib
streamlit
ipykernel
```

---

## 🔮 Future Improvements

- Automate weekly model retraining using Apache Airflow or AWS Lambda
- Add Transformer-based architecture (Temporal Fusion Transformer) for comparison
- Incorporate sentiment analysis from financial news as additional features
- Extend to multi-step forecasting (predict next 5 days instead of next 1 day)
- Add support for multiple tickers beyond AAPL

---

## 👤 Author

**Ankan**
B.Tech — Metallurgical & Materials Engineering, NIT Raipur
SIH 2025 Joint Winner | Data Analyst Intern @ Hindalco Industries

[![GitHub](https://img.shields.io/badge/GitHub-yourusername-black?logo=github)](https://github.com/yourusername)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ankan-blue?logo=linkedin)](https://linkedin.com/in/yourprofile)

---

## ⚠️ Disclaimer

This project is intended for educational and portfolio purposes only. It is not financial advice and should not be used for actual trading decisions.
