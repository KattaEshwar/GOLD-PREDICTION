from flask import Flask, request, jsonify, render_template 
import joblib 

model = joblib.load('polynomial_regression_model.pkl')

app = Flask(__name__)

@app.route('/')

def home():
    return "Polynomial Regression API is running!"  #

@app.route('/predict', methods=['POST'])

def predict():

    data = request.get_json()

    year = data.get('Year')

    if not year:
        return jsonify({'error': 'Year is required'}), 400 
    
    try:
        prediction = model.predict([[int(year+1)]])[0]

        return jsonify({
            'Year': year, 
            'Predicted Gold Price': round(prediction, 2) 
        })
    except Exception as e:

        return jsonify({'error': str(e)}), 500

@app.route('/web')
def web():
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 
