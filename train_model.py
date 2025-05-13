import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

# Read the dataset
try:
    data = pd.read_excel('Gold-Price-Prediction-Web-Application-main/DataSet.xlsx')
except:
    data = pd.read_excel('DataSet.xlsx')

# Print column names for debugging
print("Column names in the dataset:", data.columns.tolist())
print("\nFirst few rows of data:")
print(data.head())

# Prepare the data
year_col = data.columns[0]  # First column should be year
price_col = data.columns[1]  # Second column should be price

# Ensure X is a 2D array with one feature
X = data[year_col].values.reshape(-1, 1)  # This makes it a 2D array
y = data[price_col].values

print("\nShape of X:", X.shape)
print("First few values of X:", X[:5])

# Create and fit polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("\nShape of X_poly:", X_poly.shape)
print("First few values of X_poly:", X_poly[:5])

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Test prediction
test_year = np.array([[2025]])  # Test with a single year
test_poly = poly.transform(test_year)
test_pred = model.predict(test_poly)
print(f"\nTest prediction for year 2025: {test_pred[0]:.2f}")

# Save both the model and the polynomial features transformer
joblib.dump(model, 'model.pkl')
joblib.dump(poly, 'poly.pkl')

print("\nModel training completed successfully!") 