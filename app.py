# app.py
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")
st.markdown("Predict next day's closing price using a trained LSTM model")

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.header("Settings")
ticker   = st.sidebar.text_input("Stock Ticker", value="AAPL")
start    = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
end      = st.sidebar.date_input("End Date",   value=pd.to_datetime("2024-12-31"))
LOOKBACK = 60

# ── Load Model & Scaler ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model('model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# ── Load & Process Data ───────────────────────────────────────────
@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    
    # Fix MultiIndex columns
    df.columns = df.columns.get_level_values(0)
    
    # Compute indicators
    df['RSI_14']      = RSIIndicator(df['Close'], window=14).rsi()
    macd              = MACD(df['Close'])
    df['MACD']        = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb                = BollingerBands(df['Close'], window=20)
    df['BB_High']     = bb.bollinger_hband()
    df['BB_Low']      = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df

df = load_data(ticker, str(start), str(end))

# Explicitly select features in exact same order as training
features = ['Close', 'High', 'Low', 'Open', 'Volume', 
            'RSI_14', 'MACD', 'MACD_Signal', 'BB_High', 'BB_Low']

# Verify shape before scaling
data = df[features].values
st.write(f"Feature shape: {data.shape}")  # should be (n, 10)

scaled = scaler.transform(data)

# ── Create Sequences ──────────────────────────────────────────────
def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, LOOKBACK)

# ── Predictions ───────────────────────────────────────────────────
predictions = model.predict(X)

def inverse_transform(preds, scaler, n_features=10):
    dummy = np.zeros((len(preds), n_features))
    dummy[:, 0] = preds.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_pred   = inverse_transform(predictions, scaler)
y_actual = inverse_transform(y.reshape(-1,1), scaler)

# ── Metrics ───────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mae  = mean_absolute_error(y_actual, y_pred)
mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
da   = np.mean((np.diff(y_actual) > 0) == (np.diff(y_pred) > 0)) * 100

# ── Display Metrics ───────────────────────────────────────────────
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE",  f"${rmse:.2f}")
col2.metric("MAE",   f"${mae:.2f}")
col3.metric("MAPE",  f"{mape:.2f}%")
col4.metric("Directional Accuracy", f"{da:.2f}%")

# ── Plot Actual vs Predicted ──────────────────────────────────────
st.subheader("📉 Actual vs Predicted Price")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df.index[LOOKBACK:], y_actual, label='Actual',    color='black', linewidth=1)
ax.plot(df.index[LOOKBACK:], y_pred,   label='Predicted', color='blue',  linewidth=1, linestyle='--')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# ── Next Day Prediction ───────────────────────────────────────────
st.subheader("🔮 Next Day Prediction")
last_60     = scaled[-LOOKBACK:]
last_60     = last_60.reshape(1, LOOKBACK, 10)
next_pred   = model.predict(last_60)
next_price  = inverse_transform(next_pred, scaler)[0]
last_price  = df['Close'].iloc[-1]
change      = next_price - last_price
change_pct  = (change / last_price) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Last Close",       f"${last_price:.2f}")
col2.metric("Predicted Close",  f"${next_price:.2f}")
col3.metric("Expected Change",  f"${change:.2f}", f"{change_pct:.2f}%")

# ── Raw Data ──────────────────────────────────────────────────────
with st.expander("📄 View Raw Data"):
    st.dataframe(df.tail(20))
st.write("Data shape:", data.shape)
st.write("Columns:", df[features].columns.tolist())
