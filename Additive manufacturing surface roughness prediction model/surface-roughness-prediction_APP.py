

import streamlit as st
import pickle
import numpy as np

# Load the saved model and scalers
with open("gam_model.pkl", "rb") as model_file:
    saved_objects = pickle.load(model_file)

gam = saved_objects["model"]  # Load trained model
X_scaler = saved_objects["X_scaler"]  # Load X scaler
Y_scaler = saved_objects["Y_scaler"]  # Load Y scaler

# Streamlit UI
st.title("Surface roughness prediction")
st.write("This app predicts the surface roughness (Ra) of a 3D printed part based on the layer height (micron) and build angle (degree).")

# Get user input
new_lh = st.number_input("Enter the new value for lh:", min_value=0.0, step=0.1)
new_ang = st.number_input("Enter the new value for ang:", min_value=0.0, step=0.1)

if st.button("Predict"):
    # Standardize the new input
    X_new = np.array([[new_lh, new_ang]])
    X_new_scaled = X_scaler.transform(X_new)

    # Predict Ra (scaled)
    ra_new_pred_scaled = gam.predict(X_new_scaled)

    # Convert prediction back to original scale
    ra_new_pred_original = Y_scaler.inverse_transform(ra_new_pred_scaled.reshape(-1, 1))

    # Display the result
    st.success(f"Predicted Surface roughness: {ra_new_pred_original[0][0]:.4f}")