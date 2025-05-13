import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime, timedelta

# Create comprehensive dataset (monthly data from 1964 to 2100)
def generate_monthly_dates(start_year, end_year):
    dates = []
    prices = []
    
    # Base price in 1964
    base_price = 63.25
    
    # Generate monthly data
    current_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    while current_date <= end_date:
        dates.append(current_date)
        
        # Calculate price with realistic long-term growth
        years_passed = (current_date.year - start_year) + (current_date.month - 1) / 12
        
        # Add multiple growth components:
        # 1. Base exponential growth (slower for long-term stability)
        # 2. Cyclical component (7-year cycles)
        # 3. Monthly seasonal variations
        base_growth = base_price * (1.08 ** (years_passed / 2))  # Slower exponential growth
        cycle_component = 1 + 0.15 * np.sin(years_passed / 7 * 2 * np.pi)  # 7-year cycle
        seasonal_component = 1 + 0.05 * np.sin(current_date.month / 12 * 2 * np.pi)  # Monthly seasonality
        
        price = base_growth * cycle_component * seasonal_component
        prices.append(price)
        
        # Move to next month
        if current_date.month == 12:
            current_date = datetime(current_date.year + 1, 1, 1)
        else:
            current_date = datetime(current_date.year, current_date.month + 1, 1)
    
    return dates, prices

# Generate data from 1964 to 2100
dates, prices = generate_monthly_dates(1964, 2100)

# Convert dates to numerical features
def date_to_features(dates):
    features = []
    for date in dates:
        # Convert date to year and month features
        year_feature = date.year + (date.month - 1) / 12
        features.append([year_feature])
    return np.array(features)

# Prepare features
X = date_to_features(dates)

# Create polynomial features (reduced degree for more stable long-term predictions)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, prices)

# Test predictions for various future dates
test_dates = [
    datetime(2025, 1, 1),
    datetime(2030, 6, 1),
    datetime(2050, 1, 1),
    datetime(2075, 1, 1),
    datetime(2100, 12, 1)
]

print("\nTest Predictions:")
for date in test_dates:
    year_feature = date.year + (date.month - 1) / 12
    X_test = np.array([[year_feature]])
    X_test_poly = poly.transform(X_test)
    prediction = model.predict(X_test_poly)[0]
    print(f"Predicted price for {date.strftime('%B %Y')}: ₹{prediction:,.2f}")

# Save the model and transformer
joblib.dump(model, 'gold_model.pkl')
joblib.dump(poly, 'gold_poly.pkl')

print("\nModel created successfully!") 