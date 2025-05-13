import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("house_price_model.pkl")

# Streamlit UI
st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏠 House Price Predictor")
st.markdown("Estimate residential property prices based on features like rooms, floor area, and more.")

# Input fields
num_rooms = st.number_input("Number of Rooms", min_value=1, value=3)
floor_area = st.number_input("Floor Area (sq ft)", min_value=100, value=1000)
zip_code = st.text_input("ZIP Code", value="400005")
school_distance = st.number_input("Distance to Nearest School (km)", min_value=0.0, value=2.5)
crime_rate = st.number_input("Crime Rate (index)", min_value=0.0, value=2.0)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2010)

if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "num_rooms": num_rooms,
        "floor_area": floor_area,
        "zip_code": zip_code,
        "school_distance": school_distance,
        "crime_rate": crime_rate,
        "year_built": year_built
    }])

    prediction = model.predict(input_df)[0]
    inr_price = 83.00 * prediction

    st.success(f"🏷️ Predicted Price: ₹{inr_price:,.2f}")
