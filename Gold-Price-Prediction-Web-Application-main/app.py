from flask import Flask, request, render_template
import numpy as np
import joblib
from datetime import datetime

# Create Flask app
app = Flask(__name__)

# Load the model and transformer
try:
    model = joblib.load('polynomial_regression_model.pkl')
except:
    model = joblib.load('Gold-Price-Prediction-Web-Application-main/polynomial_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the year from form data
        year = int(request.form.get('year'))
        
        # Prepare input data
        X = np.array([[year]])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Return prediction
        return render_template('index.html', 
                             prediction=f"Predicted Gold Price for year {year}: ₹{prediction:,.2f}")
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', 
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # Run the app on all network interfaces
    app.run(host='0.0.0.0', port=8080, debug=True)
