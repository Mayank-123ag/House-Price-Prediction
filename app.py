import streamlit as st
import pickle
import numpy as np

model = pickle.load(
    open("model.pkl", "rb")
)

st.title("California House Price Predictor")

medinc = st.number_input("Median Income")
houseage = st.number_input("House Age")
averooms = st.number_input("Average Rooms")
avebedrms = st.number_input("Average Bedrooms")
population = st.number_input("Population")
aveoccup = st.number_input("Average Occupancy")
latitude = st.number_input("Latitude")
longitude = st.number_input("Longitude")

if st.button("Predict Price"):

    features = np.array([[
        medinc,
        houseage,
        averooms,
        avebedrms,
        population,
        aveoccup,
        latitude,
        longitude
    ]])

    prediction = model.predict(features)

    st.success(
        f"Predicted House Value: {prediction[0]:.2f}"
    )

    