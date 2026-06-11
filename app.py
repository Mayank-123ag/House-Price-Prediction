import streamlit as st
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor

@st.cache_resource
def load_model():
    housing = fetch_california_housing(as_frame=True)

    X = housing.data
    y = housing.target

    model = RandomForestRegressor(
        n_estimators=20,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    return model

model = load_model()

st.title("California House Price Predictor")

st.write("Enter housing details below:")

medinc = st.number_input("Median Income", value=3.0)
houseage = st.number_input("House Age", value=20.0)
averooms = st.number_input("Average Rooms", value=5.0)
avebedrms = st.number_input("Average Bedrooms", value=1.0)
population = st.number_input("Population", value=1000.0)
aveoccup = st.number_input("Average Occupancy", value=3.0)
latitude = st.number_input("Latitude", value=34.0)
longitude = st.number_input("Longitude", value=-118.0)

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
        f"Predicted House Value: ${prediction[0]*100000:.0f}"
    )