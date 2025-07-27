import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Title
st.title("🚗 Car Price Prediction App")

# Sidebar Inputs
st.sidebar.header("Car Features")

def user_input_features():
    fueltype = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
    aspiration = st.sidebar.selectbox("Aspiration", ["std", "turbo"])
    carbody = st.sidebar.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
    drivewheel = st.sidebar.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
    enginesize = st.sidebar.slider("Engine Size (cc)", 60, 350, 130)
    horsepower = st.sidebar.slider("Horsepower", 50, 300, 100)
    peakrpm = st.sidebar.slider("Peak RPM", 4000, 7000, 5000)
    citympg = st.sidebar.slider("City Mileage (mpg)", 10, 50, 25)
    highwaympg = st.sidebar.slider("Highway Mileage (mpg)", 10, 60, 30)

    data = {
        'fueltype': fueltype,
        'aspiration': aspiration,
        'carbody': carbody,
        'drivewheel': drivewheel,
        'enginesize': enginesize,
        'horsepower': horsepower,
        'peakrpm': peakrpm,
        'citympg': citympg,
        'highwaympg': highwaympg
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Show input
st.subheader("Input Features")
st.write(input_df)

# Encode categorical features (same way as during training)
df_encoded = pd.get_dummies(input_df)
required_cols = joblib.load("model_features.pkl")  # list of columns used in training
for col in required_cols:
    if col not in df_encoded.columns:
        df_encoded[col] = 0
df_encoded = df_encoded[required_cols]  # align columns

# Prediction
prediction = model.predict(df_encoded)[0]
st.subheader("Predicted Car Price (USD)")
st.success(f"${prediction:,.2f}")
