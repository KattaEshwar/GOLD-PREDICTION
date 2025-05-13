from flask import Flask, request, render_template
import numpy as np
import joblib
from datetime import datetime

# Create Flask app
app = Flask(__name__)

# Load the model and transformer
model = joblib.load('gold_model.pkl')
poly = joblib.load('gold_poly.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the date components from the form
        year = int(request.form.get('Year'))
        month = int(request.form.get('Month'))
        
        # Create datetime object
        prediction_date = datetime(year, month, 1)
        
        # Convert to feature (year + month/12)
        year_feature = prediction_date.year + (prediction_date.month - 1) / 12
        
        # Prepare input data
        X = np.array([[year_feature]])
        
        # Transform features and predict
        X_poly = poly.transform(X)
        prediction = model.predict(X_poly)[0]
        
        # Format the prediction
        formatted_date = prediction_date.strftime('%B %Y')
        formatted_price = "{:,.2f}".format(prediction)
        
        # Return prediction
        return render_template('index.html', 
                             prediction=f"Predicted Gold Price for {formatted_date}: ₹{formatted_price}")
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return render_template('index.html', 
                             prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    # Set host to '0.0.0.0' to make it publicly accessible
    # Set port to 80 for standard HTTP access (requires root/admin privileges)
    # For non-root users, we can use port 8080
    app.run(host='0.0.0.0', port=8080, debug=False) 