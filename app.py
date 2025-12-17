import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Human Activity Recognition", layout="centered")
st.title("🏃 Human Activity Recognition (CNN)")
st.write("Upload sensor data CSV to predict activities")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "har_cnn_model.h5")
    if not os.path.exists(model_path):
        st.error("❌ har_cnn_model.h5 NOT FOUND in project folder")
        st.stop()
    return tf.keras.models.load_model(model_path)

# -----------------------------
# Load preprocessors (scaler + label encoder + num_features)
# -----------------------------
@st.cache_resource
def load_preprocessors():
    df = pd.read_csv("Training_set.csv")

    # Activity column
    y = df["activity"]

    # Features: remove activity column & any non-numeric columns
    X = df.drop(columns=["activity"])
    X = X.select_dtypes(include=[np.number])

    # Label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y)

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    num_features = X.shape[1]

    return scaler, label_encoder, num_features

# -----------------------------
# Initialize model and preprocessors
# -----------------------------
model = load_model()
scaler, label_encoder, num_features = load_preprocessors()

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Testing CSV (same format as Training_set.csv, WITHOUT label column)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")

        # Feature validation
        if test_df.shape[1] != num_features:
            st.error(
                f"❌ Feature mismatch! Expected {num_features} columns, "
                f"but got {test_df.shape[1]}"
            )
        else:
            # Preprocess
            X_test = scaler.transform(test_df.values)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Predict
            predictions = model.predict(X_test)
            predicted_classes = np.argmax(predictions, axis=1)
            predicted_labels = label_encoder.inverse_transform(predicted_classes)

            # Output
            results_df = test_df.copy()
            results_df["Predicted Activity"] = predicted_labels

            st.subheader("✅ Prediction Results")
            st.dataframe(results_df, use_container_width=True)

            # Download button
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Predictions",
                csv,
                "har_predictions.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("CNN-based Human Activity Recognition using Streamlit")
