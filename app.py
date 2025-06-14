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
log_ex_showroom_price = 0 
city_mileage = st.number_input("City Mileage (km/l)", value=15.0)
extended_warranty = st.number_input("Extended Warranty (Years)", value=2.0)
power = st.number_input("Power (PS)", value=100.0)
seating_capacity = st.number_input("Seating Capacity", value=5)

log_displacement = st.number_input("log(Displacement)", value=7.5)
log_height = st.number_input("log(Height)", value=5.4)
log_length = st.number_input("log(Length)", value=6.5)
log_width = st.number_input("log(Width)", value=5.7)
log_price = 0  # dummy, not needed for prediction
make = st.selectbox("Make", [
    'Maruti Suzuki', 'Hyundai', 'Honda', 'Ford', 'Fiat', 'Audi', 'Tata', 
    'Skoda', 'Volkswagen', 'Porsche', 'Nissan', 'Mahindra', 'Land Rover Rover',
    'Renault', 'Datsun', 'Bentley', 'Maserati', 'Aston Martin'
])
model_name = st.selectbox("Model", [
    'A3', 'Accord Hybrid', 'Flying Spur', 'Kodiaq', 'Kuv100 Nxt',
    'Linea Classic', 'Monte Carlo', 'NULL', 'Range Velar', 'Redi-Go',
    'Superb Sportline', 'Tuv300 Plus', 'Verito Vibe', 'Avventura', 'Bolt',
    'Celerio', 'City', 'Duster', 'Dzire', 'Elantra', 'Endeavour', 'Go',
    'Grancabrio', 'Granturismo', 'Jazz', 'Levante', 'Linea', 'Panamera',
    'Polo', 'Q3', 'Q5', 'Q7', 'Quattroporte', 'Range', 'Rapid', 'Rapide',
    'Rs7', 'Sunny', 'Superb', 'Terrano', 'Tiago', 'Tigor', 'Tuv300',
    'Vento', 'Xuv500', 'Zest'
])


emission_norm = st.selectbox("Emission Norm", ['BS IV', 'BS 6'])

ventilation_system = st.selectbox("Ventilation System", [
    "2 Zone Climate Control",
    "3 Zone climate control",
    "4 Zone climate control",
    "Fully automatic climate control",
    "Fully automatic climate control, 2 Zone Climate Control",
    "Fully automatic climate control, 4 Zone climate control",
    "Heater, Manual Air conditioning with cooling and heating",
    "Manual Air conditioning with cooling and heating"
])

abs_option = st.selectbox("ABS", ['Yes', 'No'], key="abs")
ebd_option = st.selectbox("EBD", ['Yes', 'No'], key="ebd")
average_fuel_consumption = st.selectbox("Average Fuel Consumption", ['Low', 'Moderate', 'High'], key="avg_fuel")
central_locking = st.selectbox("Central Locking", ['Yes', 'No'], key="central_locking")
child_safety_locks = st.selectbox("Child Safety Locks", ['Yes', 'No'], key="child_locks")
engine_malfunction_light = st.selectbox("Engine Malfunction Light", ['Yes', 'No'], key="malfunction")
front_brakes = st.selectbox("Front Brakes", ['Disc', 'Drum'], key="brakes")
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'CNG'], key="fuel_type")
engine_immobilizer = st.selectbox("Engine Immobilizer", ['Yes', 'No'], key="immobilizer")



# Create a single-row dataframe
input_dict = {
    'log_ex_showroom_price': [log_ex_showroom_price],
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
    'model':[model_name],
    'emission_norm':[emission_norm],
    'ventilation_system':[ventilation_system],
     'abs': [abs_option],
     'ebd': [ebd_option],
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
