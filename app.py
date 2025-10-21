import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from tensorflow.keras.models import load_model/Users/macbook/Desktop/internship/app.py
import joblib
import os

st.set_page_config(page_title="🌍 Forex Forecasting Dashboard", layout="centered")
st.title("🌍 Forex Forecasting Dashboard")
st.write("Select a currency and forecast horizon below to generate predictions.")

# --- Load saved models ---
@st.cache_resource
def load_all_models():
    models = {}
    model_folder = "models"  # <-- folder where you saved your models
    if not os.path.exists(model_folder):
        st.warning("⚠️ 'models' folder not found. Place your saved models inside a folder named 'models'.")
        return models

    for filename in os.listdir(model_folder):
        filepath = os.path.join(model_folder, filename)
        if filename.endswith(".pkl"):  # Prophet, ARIMA, LightGBM, XGBoost, etc.
            model = joblib.load(filepath)
            model_type = "Prophet" if "prophet" in filename.lower() else "Other"
        elif filename.endswith(".h5"):  # LSTM or deep learning models
            model = load_model(filepath)
            model_type = "LSTM"
        else:
            continue
        models[filename.replace(".pkl", "").replace(".h5", "")] = {
            "model": model,
            "type": model_type
        }
    return models


models = load_all_models()

# --- Handle no models found ---
if len(models) == 0:
    st.error("❌ No models loaded. Please ensure your 'models' folder contains .pkl or .h5 files.")
else:
    countries = list(models.keys())
    selected_country = st.selectbox("🌐 Select a Currency", countries)
    model_info = models[selected_country]
    model_type = model_info["type"]

    st.write(f"**Model loaded for {selected_country}:** {model_type}")

    # --- Forecast horizon ---
    horizon = st.number_input("📅 Forecast horizon (days)", min_value=1, max_value=365, value=30)

    # --- Generate forecast button ---
    if st.button("🚀 Generate Forecast"):
        try:
            model = model_info["model"]

            # --- Forecast function ---
            def make_forecast(model_type, model, horizon):
                if model_type == "Prophet":
                    # Prophet expects a dataframe with 'ds'
                    future = model.make_future_dataframe(periods=horizon)
                    forecast = model.predict(future)
                    forecast = forecast[['ds', 'yhat']]
                    forecast.rename(columns={'ds': 'Date', 'yhat': 'Forecast'}, inplace=True)

                elif model_type == "LSTM":
                    input_data = np.random.rand(30, 1).reshape(-1, 1)
                    pred = model.predict(input_data)
                    forecast = pd.DataFrame({
                        "Date": pd.date_range(start=pd.Timestamp.today(), periods=len(pred)),
                        "Forecast": pred.flatten()
                    })

                else:  # Other ML models
                    input_data = np.random.rand(horizon, 1)
                    pred = model.predict(input_data)
                    forecast = pd.DataFrame({
                        "Date": pd.date_range(start=pd.Timestamp.today(), periods=len(pred)),
                        "Forecast": pred
                    })
                return forecast

            forecast = make_forecast(model_type, model, horizon)

            # --- Display forecast ---
            st.subheader(f"📊 Forecast for {selected_country}")
            st.dataframe(forecast)
            st.line_chart(forecast.set_index("Date")["Forecast"])

        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")
