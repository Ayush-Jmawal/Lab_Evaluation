import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Car Price & Depreciation Predictor", layout="wide")

st.title("🚗 Car Price Predictor + 📉 Depreciation Visualizer")

# --- Train and Save Model (first time use only) ---
@st.cache_resource
def train_model():
    df = pd.read_csv("CarPrice.csv")

    df.drop(["CarName", "car_ID"], axis=1, inplace=True)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_encoded = pd.get_dummies(X)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_encoded, y)

    joblib.dump(model, "model.pkl")
    joblib.dump(X_encoded.columns.tolist(), "model_features.pkl")

    return model, X_encoded.columns.tolist()

# Load model and feature columns
try:
    model = joblib.load("model.pkl")
    model_features = joblib.load("model_features.pkl")
except:
    model, model_features = train_model()

# --- Input Section ---
st.sidebar.header("Enter Car Features")

symboling = st.sidebar.slider("Symboling", -3, 3, 0)
fueltype = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.sidebar.selectbox("Aspiration", ["std", "turbo"])
doornumber = st.sidebar.selectbox("Number of Doors", ["two", "four"])
carbody = st.sidebar.selectbox("Car Body", ["convertible", "hatchback", "sedan", "wagon", "hardtop"])
drivewheel = st.sidebar.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
enginelocation = st.sidebar.selectbox("Engine Location", ["front", "rear"])
wheelbase = st.sidebar.slider("Wheelbase", 80.0, 120.0, 95.0)
carlength = st.sidebar.slider("Car Length", 140.0, 210.0, 170.0)
carwidth = st.sidebar.slider("Car Width", 60.0, 75.0, 65.0)
carheight = st.sidebar.slider("Car Height", 48.0, 60.0, 52.0)
curbweight = st.sidebar.slider("Curb Weight", 1500, 4000, 2500)
enginetype = st.sidebar.selectbox("Engine Type", ["dohc", "ohcv", "ohc", "l", "rotor", "ohcf", "dohcv"])
cylindernumber = st.sidebar.selectbox("Number of Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"])
enginesize = st.sidebar.slider("Engine Size", 60, 350, 150)
fuelsystem = st.sidebar.selectbox("Fuel System", ["mpfi", "2bbl", "1bbl", "4bbl", "idi", "spdi", "mfi", "spfi"])
boreratio = st.sidebar.slider("Bore Ratio", 2.5, 4.0, 3.0)
stroke = st.sidebar.slider("Stroke", 2.0, 4.5, 3.0)
compressionratio = st.sidebar.slider("Compression Ratio", 7.0, 23.0, 9.0)
horsepower = st.sidebar.slider("Horsepower", 50, 300, 100)
peakrpm = st.sidebar.slider("Peak RPM", 4000, 7000, 5000)
citympg = st.sidebar.slider("City MPG", 10, 50, 25)
highwaympg = st.sidebar.slider("Highway MPG", 15, 60, 30)

# --- Create Input DataFrame ---
input_dict = {
    "symboling": symboling,
    "wheelbase": wheelbase,
    "carlength": carlength,
    "carwidth": carwidth,
    "carheight": carheight,
    "curbweight": curbweight,
    "enginesize": enginesize,
    "boreratio": boreratio,
    "stroke": stroke,
    "compressionratio": compressionratio,
    "horsepower": horsepower,
    "peakrpm": peakrpm,
    "citympg": citympg,
    "highwaympg": highwaympg,
    "fueltype_" + fueltype: 1,
    "aspiration_" + aspiration: 1,
    "doornumber_" + doornumber: 1,
    "carbody_" + carbody: 1,
    "drivewheel_" + drivewheel: 1,
    "enginelocation_" + enginelocation: 1,
    "enginetype_" + enginetype: 1,
    "cylindernumber_" + cylindernumber: 1,
    "fuelsystem_" + fuelsystem: 1,
}

# Create DataFrame and fill missing features with 0
input_data = pd.DataFrame([input_dict])
for col in model_features:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model_features]

# --- Prediction ---
prediction = model.predict(input_data)[0]

st.subheader("🔍 Estimated Car Price:")
st.success(f"${prediction:,.2f}")

# --- Depreciation Visualizer ---
st.subheader("📉 Depreciation Over Time")

depreciation_rate = st.slider("Estimated Annual Depreciation Rate (%)", 5, 25, 15)
years = np.arange(0, 11)
depreciated_prices = [prediction * ((1 - depreciation_rate / 100) ** y) for y in years]

fig, ax = plt.subplots()
ax.plot(years, depreciated_prices, marker='o', color='orange')
ax.set_title("Car Depreciation Over 10 Years")
ax.set_xlabel("Years")
ax.set_ylabel("Estimated Value ($)")
ax.grid(True)
st.pyplot(fig)

st.caption("Note: This is a simple linear depreciation model. Actual values vary by brand, market, and condition.")
