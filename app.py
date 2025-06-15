# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('model.pkl')

st.title("Car Resale Price Predictor")

# Input fields
st.subheader("Enter Car Details:")

city_mileage = st.number_input("City Mileage (km/l)", value=15.0)
extended_warranty = st.number_input("Extended Warranty (Years)", value=2.0)
power = st.number_input("Power (PS)", value=100.0)
seating_capacity = st.number_input("Seating Capacity", value=5)

log_displacement = st.number_input("log(Displacement)", value=7.5)
log_height = st.number_input("log(Height)", value=5.4)
log_length = st.number_input("log(Length)", value=6.5)
log_width = st.number_input("log(Width)", value=5.7)
log_price = 0  # dummy, not needed for prediction
make = st.selectbox("Make", ['Maruti', 'Hyundai', 'Honda'])  # You can update these options
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
central_locking = st.selectbox("Central Locking", ['Yes', 'No'])
model_name = st.selectbox("Model", ['Nano Genx', 'Redi-Go', 'Kwid','Alto K10'])
emission_norm = st.selectbox("Emission Norm", ['BS IV', 'BS 6','BS III'])
ventilation_system = st.selectbox("Ventilation System", ['Yes', 'Manual Air conditioning with cooling and heating','NO'])
average_fuel_consumption = st.selectbox("Average Fuel Consumption", ['YES', 'NO'])
child_safety_locks = st.selectbox("Child Safety Locks", ['Yes', 'No'])
engine_malfunction_light = st.selectbox("Engine Malfunction Light", ['Yes', 'No'])
front_brakes = st.selectbox("Front Brakes", ['ventilatedDisk', 'Drum','Solid Disc'])
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'])
engine_immobilizer = st.selectbox("Engine Immobilizer", ['Yes', 'No'])


# Create a single-row dataframe
input_dict = {
    'city_mileage_numeric': [city_mileage],
    'extended_warranty_numeric': [extended_warranty],
    'power_numeric': [power],
    'seating_capacity': [seating_capacity],
    'log_displacement': [log_displacement],
    'log_height': [log_height],
    'log_length': [log_length],
    'log_width': [log_width],
    'make': [make],
    'fuel_type': [fuel_type],
    'central_locking': [central_locking],
    'model':[model],
    'emission_norm':[emission_norm],
    'ventilation_system':[ventilation_system],
    'average_fuel_consumption':[average_fuel_consumption],
    'child_safety_locks':[child_safety_locks],
    'engine_malfunction_light':[engine_malfunction_light],
    'front_brakes':[front_brakes],
    'fuel_type':[fuel_type],
    'engine_immobilizer':[engine_immobilizer]
}

input_df = pd.DataFrame(input_dict)

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Resale Price: ₹ {prediction:,.2f}")
