import streamlit as st
import pandas as pd
import pickle
import os
import joblib

#Define the model file path
MODEL_FILE_PATH = 'regression_model.pkl'

#Load the saved model
with open(MODEL_FILE_PATH, 'rb') as f:
    model = joblib.load(f)

#Create a Streamlit app
st.title("CO2 Emissions Prediction")

#Create a form to input features
with st.form("prediction_form"):
    engine_size = st.number_input("Engine Size")
    cylinders = st.number_input("Cylinders")
    fuel_consumption_city = st.number_input("Fuel Consumption City")
    fuel_consumption_hw = st.number_input("Fuel Consumption HW")
    fuel_consumption_comb = st.number_input("Fuel Consumption Comb")
    fuel_consumption_comb_mpg = st.number_input("Fuel Consumption Comb MPG")

    # Create a submit button
    submitted = st.form_submit_button("predict")

    # Make a prediction when the form is submitted
    if submitted:
        features = pd.DataFrame([[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hw, fuel_consumption_comb, fuel_consumption_comb_mpg]], columns=[ 'ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_COMB_MPG'])
        prediction = model.predict(features)
        st.write("Predicted CO2 Emissions:", prediction[0])
