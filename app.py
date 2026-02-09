import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Iris Flower Prediction", layout="centered")

st.title("🌸 Iris Flower Prediction Project")

st.write("This app predicts the species of an Iris flower using Machine Learning.")

# Debug: show current directory
st.write("📂 Current working directory:")
st.code(os.getcwd())

try:
    # Load model and encoder
    model = joblib.load("iris_model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    st.success("✅ Model and Encoder loaded successfully")

    # User inputs
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2)

    if st.button("🔍 Predict"):
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)
        species = encoder.inverse_transform(prediction)
        st.success(f"🌼 Predicted Species: **{species[0]}**")

except Exception as e:
    st.error("❌ Application Error")
    st.exception(e)

