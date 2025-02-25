import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from pygam import GammaGAM

data = pd.read_csv('dataset.csv')
data.head()
#NOTE layer height is in microns, build angle is in degree and surface roughness is in microns

#Plot surface roughness vs build angle for each layer height
from matplotlib import pyplot as plt
data.plot(kind='scatter', x='build angle', y='surface roughness', s=32, alpha=.8)
plt.gca().spines[['top', 'right',]].set_visible(False)

# Convert DataFrame columns to NumPy arrays explicitly
X = data[['layer height', 'build angle']].values  # Convert to NumPy array
Y = data['surface roughness'].values.reshape(-1, 1)  # Ensure Y is 2D

# Define separate scalers for X and Y
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit the scalers on training data
X_scaled = X_scaler.fit_transform(X)
Y_scaled = Y_scaler.fit_transform(Y)


# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

# Train the model
gam = GammaGAM().fit(X_train, y_train)

from sklearn.metrics import r2_score

# Predict on test data
y_test_pred = gam.predict(X_test)

# Compute R² Score
r2 = r2_score(y_test, y_test_pred)

# Compute Adjusted R² Score
n = X_test.shape[0]  # Number of test samples
p = X_test.shape[1]  # Number of predictor variables

adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

# Print the scores
print("R² Score:", r2)
print("Adjusted R² Score:", adjusted_r2)


# Get new user inputs
new_layer_height = float(input("Enter the new value for layer height in micron: "))
new_build_angle = float(input("Enter the new value for build angle in degree: "))

# Convert new input values into a NumPy array
X_new = np.array([[new_layer_height, new_build_angle]])

# Standardize new input
X_new_scaled = X_scaler.transform(X_new)

# Predict scaled output
ra_new_pred_scaled = gam.predict(X_new_scaled)

# Inverse transform prediction
ra_new_pred_original = Y_scaler.inverse_transform(ra_new_pred_scaled.reshape(-1, 1))

print("Predicted Surface roughness in micron:", ra_new_pred_original[0][0])

import pickle

# Save the trained model and scalers
with open("gam_model.pkl", "wb") as model_file:
    pickle.dump({
        "model": gam,  # The trained GammaGAM model
        "X_scaler": X_scaler,  # Scaler for input features
        "Y_scaler": Y_scaler   # Scaler for output
    }, model_file)

print("Model saved successfully")