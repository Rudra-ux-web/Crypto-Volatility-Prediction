from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Trained Model
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "Notebook" / "models" / "random_forest_model.pkl"

model = joblib.load(MODEL_PATH)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="Cryptocurrency Volatility Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("Cryptocurrency Volatility Prediction")
st.write("Enter cryptocurrency details below and click **Predict**.")

st.divider()

# -------------------------------
# User Inputs
# -------------------------------
open_price = st.number_input("Open Price", min_value=0.0, value=100.0)

high = st.number_input("High Price", min_value=0.0, value=105.0)

low = st.number_input("Low Price", min_value=0.0, value=95.0)

close = st.number_input("Close Price", min_value=0.0, value=102.0)

volume = st.number_input("Trading Volume", min_value=0.0, value=1000000.0)

marketCap = st.number_input("Market Capitalization", min_value=1.0, value=50000000.0)

st.divider()

# -------------------------------
# Predict Button
# -------------------------------
if st.button("Predict Volatility"):

    # Feature Engineering
    daily_return = (close - open_price) / open_price if open_price != 0 else 0

    price_range = high - low

    liquidity_ratio = volume / marketCap if marketCap != 0 else 0

    MA7 = close

    MA30 = close

    rolling_volatility = 0

    lag_close = close

    # Create DataFrame
    input_data = pd.DataFrame([[
        open_price,
        high,
        low,
        close,
        volume,
        marketCap,
        daily_return,
        price_range,
        liquidity_ratio,
        MA7,
        MA30,
        rolling_volatility,
        lag_close
    ]], columns=[
        "open",
        "high",
        "low",
        "close",
        "volume",
        "marketCap",
        "daily_return",
        "price_range",
        "liquidity_ratio",
        "MA7",
        "MA30",
        "rolling_volatility",
        "lag_close"
    ])

    prediction = model.predict(input_data)

    st.success(f"✅ Predicted Volatility: {prediction[0]:.6f}")

    st.info(f"""
### Calculated Features

- Daily Return: **{daily_return:.6f}**
- Price Range: **{price_range:.4f}**
- Liquidity Ratio: **{liquidity_ratio:.6f}**
""")