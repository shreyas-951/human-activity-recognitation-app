import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf

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
    return tf.keras.models.load_model("har_cnn_model.h5")

model = load_model()

# -----------------------------
# Fit scaler & label encoder
# -----------------------------
@st.cache_resource
def load_preprocessors():
    train_df = pd.read_csv("Training_set.csv")

    X = train_df.iloc[:, :-1].values
    y = train_df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    le = LabelEncoder()
    le.fit(y)

    return scaler, le, X.shape[1]

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
        # Read file
        test_df = pd.read_csv(uploaded_file)
        st.success("CSV uploaded successfully!")

        # Check feature count
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
            st.dataframe(results_df)

            # Download button
            csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions",
                data=csv,
                file_name="har_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("CNN-based Human Activity Recognition using Streamlit")
