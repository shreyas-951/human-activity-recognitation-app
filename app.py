import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Human Activity Recognition", layout="centered")
st.title("🏃 Human Activity Recognition (CNN)")
st.write("Upload sensor CSV to predict activities")

# -----------------------------
# Load trained model and preprocessors
# -----------------------------
model = tf.keras.models.load_model("har_cnn_model.h5")

# Load scaler and label encoder if you have saved them
# Example:
# import joblib
# scaler = joblib.load("scaler.save")
# label_encoder = joblib.load("label_encoder.save")

# For demonstration, create dummy scaler and encoder
scaler = StandardScaler()
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['sitting','standing','walking','running'])  # example classes

# -----------------------------
# Upload CSV
# -----------------------------
uploaded_file = st.file_uploader("Upload your sensor CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # Make sure you select only the column your model was trained on
    # For example, 'acc_x'
    if 'acc_x' not in df.columns:
        st.error("CSV must contain 'acc_x' column")
    else:
        X_new = df[['acc_x']].values  # shape (num_samples, 1)
        
        # Apply scaler if used during training
        X_new = scaler.fit_transform(X_new)  # replace fit_transform with transform if scaler is pre-fitted
        
        # Reshape to match model input: (batch_size, timesteps, features)
        X_new_correct = X_new.reshape(-1, 1, 1)
        
        # Predict
        y_pred = model.predict(X_new_correct)
        
        # Convert predicted probabilities to labels if using softmax
        if y_pred.shape[1] > 1:
            y_classes = np.argmax(y_pred, axis=1)
            y_labels = label_encoder.inverse_transform(y_classes)
        else:
            y_labels = (y_pred > 0.5).astype(int)
        
        # Display results
        st.write("Predicted Activities:")
        st.write(y_labels)
