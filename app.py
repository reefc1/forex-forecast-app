import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from keras.models import load_model

# --- Streamlit page config ---
st.set_page_config(page_title="🌍 Forex Forecasting Dashboard", layout="centered")
st.title("🌍 Forex Forecasting Dashboard")
st.write("Select a currency and forecast horizon below to generate predictions.")

# --- Load saved models ---
@st.cache_resource
def load_all_models():
    models = {}
    model_folder = "models"  # Folder where your models are saved
    if not os.path.exists(model_folder):
        st.warning("⚠️ 'models' folder not found. Place your saved models inside a folder named 'models'.")
        return models

    for filename in os.listdir(model_folder):
        filepath = os.path.join(model_folder, filename)
        if filename.endswith(".pkl"):  # Prophet, ARIMA, LightGBM, XGBoost, etc.
            model = joblib.load(filepath)
            model_type = "Prophet" if "prophet" in filename.lower() else "Other"
        elif filename.endswith(".h5"):  # LSTM
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
    # --- Create clean display names ---
    display_to_file = {}
    for filename in models.keys():
        clean_name = filename
        for suffix in ["_PROPHET", "_LSTM", "_XGBoost", "_LIGHTGBM"]:
            if clean_name.upper().endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        clean_name = clean_name.replace("-", " - ").title()
        display_to_file[clean_name] = filename

    # --- Dropdown with clean names ---
    selected_country_display = st.selectbox("🌐 Select a Currency", list(display_to_file.keys()))
    selected_country_file = display_to_file[selected_country_display]

    model_info = models[selected_country_file]
    model_type = model_info["type"]

    st.write(f"**Model loaded for {selected_country_display}:** {model_type}")

    # --- Forecast horizon ---
    horizon = st.number_input("📅 Forecast horizon (days)", min_value=1, max_value=365, value=30)

    # --- Forecast function ---
    def make_forecast(model_type, model, horizon):
        if model_type == "Prophet":
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

        else:  # Other ML models (XGB, LightGBM, ARIMA, etc.)
            input_data = np.random.rand(horizon, 1)
            pred = model.predict(input_data)
            forecast = pd.DataFrame({
                "Date": pd.date_range(start=pd.Timestamp.today(), periods=len(pred)),
                "Forecast": pred
            })

        return forecast

    # --- Generate forecast button ---
    if st.button("🚀 Generate Forecast"):
        try:
            model = model_info["model"]
            forecast = make_forecast(model_type, model, horizon)

            # --- Display forecast ---
            st.subheader(f"📊 Forecast for {selected_country_display}")
            st.dataframe(forecast)
            st.line_chart(forecast.set_index("Date")["Forecast"])

        except Exception as e:
            st.error(f"❌ Error generating forecast: {e}")


