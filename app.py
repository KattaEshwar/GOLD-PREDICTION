from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

# Load the model
try:
    model = joblib.load('polynomial_regression_model.pkl')
    logger.info("Model loaded successfully")
except Exception as e:
    try:
        model = joblib.load('Gold-Price-Prediction-Web-Application-main/polynomial_regression_model.pkl')
        logger.info("Model loaded successfully from alternate path")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date components from the form
        year = int(request.form.get('Year'))
        month = int(request.form.get('Month'))
        
        logger.info(f"Prediction requested for Year: {year}, Month: {month}")
        
        # Create datetime object
        prediction_date = datetime(year, month, 1)
        
        # Convert to feature (year + month/12)
        year_feature = prediction_date.year + (prediction_date.month - 1) / 12
        
        # Prepare input data as a DataFrame-like structure
        X = np.array([[year_feature]])
        
        # The model is a pipeline that includes PolynomialFeatures, so we can use it directly
        prediction = model.predict(X)[0]
        
        # Format the prediction
        formatted_date = f"{month}/{year}"
        formatted_price = "{:,.2f}".format(prediction)
        
        logger.info(f"Prediction successful: {formatted_price} for {formatted_date}")
        
        # Return prediction
        return render_template('index.html', 
                             prediction=f"Predicted Gold Price for {formatted_date}: ₹{formatted_price}")
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', 
                             prediction=f"Error: {str(e)}")

from waitress import serve

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)
