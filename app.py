# app.py
import streamlit as st
import pickle
import pandas as pd
import random

# Load the saved model, scaler, and label encoder
with open('linear_svm_model.pkl', 'rb') as f:
    data = pickle.load(f)

model = data['model']
scaler = data['scaler']
le = data['label_encoder']

# Streamlit page config
st.set_page_config(page_title="Iris Flower Classifier 🌸", page_icon="🌼", layout="centered")

# Overall pastel-themed background
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #FFF1E6, #E6F7FF); /* soft pastel gradient */
    }
    .stSlider > div > div {
        background-color: #FFFFFF; /* white slider container */
        border-radius: 12px;
        padding: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Iris Flower Species Prediction 🌸")
st.write("Enter the flower's measurements below to predict its species.")

# Function to create a boxed slider with soft pastel background
def boxed_slider(label, min_val, max_val, default_val, box_color):
    st.markdown(
        f"""
        <div style="padding:12px; border:2px solid {box_color}; border-radius:12px; background-color:{box_color}33; margin-bottom:10px;">
        <b>{label}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    return st.slider(label, min_val, max_val, default_val)

# Sliders with pastel colored boxes
sepal_length = boxed_slider("Sepal Length (cm)", 4.0, 8.0, 5.0, "#FFB3B3")  # pastel red
sepal_width  = boxed_slider("Sepal Width (cm)", 2.0, 4.5, 3.0, "#B3D9FF")   # pastel blue
petal_length = boxed_slider("Petal Length (cm)", 1.0, 7.0, 4.0, "#B3FFB3")  # pastel green
petal_width  = boxed_slider("Petal Width (cm)", 0.1, 2.5, 1.0, "#FFD9B3")   # pastel orange

# Feature names for DataFrame
feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']

# Predict button
if st.button("Predict Species 🌼"):
    # Prepare input as DataFrame
    X_new = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=feature_names)
    X_scaled = scaler.transform(X_new)

    # Predict
    pred_encoded = model.predict(X_scaled)
    pred_class = le.inverse_transform(pred_encoded)[0]

    # Color-coded output box
    color_dict = {
        'Iris-setosa': '#FF9999',        # pastel red
        'Iris-versicolor': '#99FF99',    # pastel green
        'Iris-virginica': '#99CCFF'      # pastel blue
    }

    st.markdown(
        f"""
        <div style="padding:20px; border:2px solid {color_dict[pred_class]}; border-radius:15px; background-color:{color_dict[pred_class]}33; text-align:center; margin-top:15px;">
        <h2>🌸 Predicted Species: {pred_class} 🌸</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()

    # Emoji celebration with species
    emoji_dict = {
        'Iris-setosa': '🌹',
        'Iris-versicolor': '🌻',
        'Iris-virginica': '🌺'
    }
    st.write(f"Celebrate! Your Iris Flower is a **{pred_class}** {emoji_dict[pred_class]}")

# Footer credit
st.markdown(
    """
    <hr>
    <p style='text-align:center; color:#555;'>Created by Sunmathi 🌸</p>
    """,
    unsafe_allow_html=True
)
