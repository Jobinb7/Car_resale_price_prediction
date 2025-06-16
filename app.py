import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('model.pkl')

# UI
st.title("Car Resale Price Predictor")
st.subheader("Enter Car Details")

# Numerical raw inputs
city_mileage = st.number_input("City Mileage (km/l)", min_value=1.0, max_value=50.0, value=15.0)
extended_warranty = st.number_input("Extended Warranty (Years)", min_value=0.0, max_value=10.0, value=2.0)
power = st.number_input("Power (PS)", min_value=30.0, max_value=1000.0, value=100.0)
seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=10, value=5)

# Raw dimension inputs
displacement = st.number_input("Displacement (cc)", min_value=500.0, max_value=10000.0, value=1500.0)
height = st.number_input("Height (mm)", min_value=1000.0, max_value=3000.0, value=1500.0)
length = st.number_input("Length (mm)", min_value=2500.0, max_value=6000.0, value=4500.0)
width = st.number_input("Width (mm)", min_value=1200.0, max_value=2500.0, value=1700.0)

# Categorical features
make = st.selectbox("Make", [
    'Maruti Suzuki', 'Hyundai', 'Honda', 'Ford', 'Fiat', 'Audi', 'Tata', 
    'Skoda', 'Volkswagen', 'Porsche', 'Nissan', 'Mahindra', 'Land Rover Rover',
    'Renault', 'Datsun', 'Bentley', 'Maserati', 'Aston Martin'
],key ="make")

model_name = st.selectbox("Model", [
    'A3', 'Accord Hybrid', 'Flying Spur', 'Kodiaq', 'Kuv100 Nxt',
    'Linea Classic', 'Monte Carlo', 'Range Velar', 'Redi-Go',
    'Superb Sportline', 'Tuv300 Plus', 'Verito Vibe', 'Avventura', 'Bolt',
    'Celerio', 'City', 'Duster', 'Dzire', 'Elantra', 'Endeavour', 'Go',
    'Grancabrio', 'Granturismo', 'Jazz', 'Levante', 'Linea', 'Panamera',
    'Polo', 'Q3', 'Q5', 'Q7', 'Quattroporte', 'Range', 'Rapid', 'Rapide',
    'Rs7', 'Sunny', 'Superb', 'Terrano', 'Tiago', 'Tigor', 'Tuv300',
    'Vento', 'Xuv500', 'Zest'
],"model")

emission_norm = st.selectbox("Emission Norm", ['BS IV', 'BS 6'],key = "emission_norm")

ventilation_system = st.selectbox("Ventilation System", [
    "2 Zone Climate Control",
    "3 Zone climate control",
    "4 Zone climate control",
    "Fully automatic climate control",
    "Fully automatic climate control, 2 Zone Climate Control",
    "Fully automatic climate control, 4 Zone climate control",
    "Heater, Manual Air conditioning with cooling and heating",
    "Manual Air conditioning with cooling and heating"
],key = "ventilation_system")
abs_option = st.selectbox("ABS", ['Yes', 'No'], key="abs")
ebd_option = st.selectbox("EBD", ['Yes', 'No'], key="ebd")

average_fuel_consumption = st.selectbox("Average Fuel Consumption", ['Low', 'Moderate', 'High'], key="average_fuel_consumption'")
central_locking = st.selectbox("Central Locking", ['Yes', 'No'], key="central_locking")
child_safety_locks = st.selectbox("Child Safety Locks", ['Yes', 'No'], key="child_locks")
engine_malfunction_light = st.selectbox("Engine Malfunction Light", ['Yes', 'No'], key="engine_malfunction_light")
front_brakes = st.selectbox("Front Brakes", ['Disc', 'Drum'], key="brakes")
fuel_type = st.selectbox("Fuel Type", ['Petrol', 'Diesel', 'Hybrid'], key="fuel_type")
engine_immobilizer = st.selectbox("Engine Immobilizer", ['Yes', 'No'], key=" engine_immobilizer")

# Log-transform required features
log_displacement = np.log1p(displacement)
log_height = np.log1p(height)
log_length = np.log1p(length)
log_width = np.log1p(width)

# Construct DataFrame
input_data = pd.DataFrame([{
    'city_mileage_numeric': city_mileage,
    'extended_warranty_numeric': extended_warranty,
    'power_numeric': power,
    'seating_capacity': seating_capacity,
    'log_displacement': log_displacement,
    'log_height': log_height,
    'log_length': log_length,
    'log_width': log_width,
    'make': make,
    'model': model_name,
    'emission_norm': emission_norm,
    'ventilation_system': ventilation_system,
    'abs': abs_option,
    'ebd': ebd_option,
    'average_fuel_consumption': average_fuel_consumption,
    'central_locking': central_locking,
    'child_safety_locks': child_safety_locks,
    'engine_malfunction_light': engine_malfunction_light,
    'front_brakes': front_brakes,
    'fuel_type': fuel_type,
    'engine_immobilizer': engine_immobilizer
}])

# Prediction
if st.button("Predict Resale Price"):
    try:
        resale_price = model.predict(input_data)[0]
        predicted_price = max(0, predicted_price) 
        st.success(f"Estimated Resale Price: ₹ {resale_price:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.write("Input DataFrame used:")
        st.write(input_data)
