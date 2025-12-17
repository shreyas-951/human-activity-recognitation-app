import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # for loading saved scaler/encoder if used

# -----------------------------
# Load model and preprocessors
# -----------------------------
@st.cache_data
def load_model_and_preprocessors():
    model = tf.keras.models.load_model("har_cnn_model.h5")
    scaler = joblib.load("scaler.save")       # optional, if you used StandardScaler
    label_encoder = joblib.load("encoder.save")  # optional, if you encoded labels
    return model, scaler, label_encoder

model, scaler, label_encoder = load_model_and_preprocessors()

st.title("🏃 Human Activity Recognition (Single Prediction)")

# -----------------------------
# User input
# -----------------------------
# Example: if your model expects 1 feature per timestep
user_input = st.text_input("Enter feature values separated by comma (e.g., 0.12):")

if st.button("Predict Activity"):
    try:
        # Convert input string to numpy array
        data = np.array([float(x.strip()) for x in user_input.split(",")])
        
        # Ensure input is 1D
        data = data.reshape(1, -1)  # shape (1, num_features)
        
        # Scale input if scaler was used
        if scaler:
            data = scaler.transform(data)
        
        # Reshape to model expected shape (batch_size, timesteps, features)
        # Adjust these numbers to match your model input
        data = data.reshape(1, 1, 1)  # (1 sample, 1 timestep, 1 feature)

        # Predict
        pred_probs = model.predict(data)
        pred_class = np.argmax(pred_probs, axis=1)
        
        # Decode label if LabelEncoder was used
        if label_encoder:
            activity = label_encoder.inverse_transform(pred_class)[0]
        else:
            activity = str(pred_class[0])
        
        st.success(f"Predicted Activity: {activity}")

    except Exception as e:
        st.error(f"Error: {e}")
