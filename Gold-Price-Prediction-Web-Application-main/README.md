# Gold Price Prediction using Flask
## Overview
This Project is a **Gold Price Prediction Web Application** built using **Flask** for the backend, a machine model for prediction, and a simple frontend for user interaction. The model predicts gold prices based on historical data and relevant economic indicators.
## Features
- Web-based interface for prediction gold prices
- Machine learning model trained on historical gold price data
- Flask backend to handle API requests
- Interactive UI for user input and prediction results
- Deployed locally
## Technology Used
- **Fronted**: HTML, CSS, Javascript # VScode
- **Backend**: Flask(Python) #VScode
- **Machine Learning**: Scikit-learn, Pandas, NumPy. # Google Colab
- **Algoritmes**: Polynomial regression
- **Deployment**: Flask server
## DataSet
DataSet contains two coloums:
1. Year
2. Corresponding price in RS.
```
DataSet.xlsx
```
## Installation and Setup
### Prerequisites
Ensure you have the followig installed:
- Python 3.7+
- Pip(Python package manager)
- Virtual environment (Optional but recommended)
### Steps to run locally
1. **Clone the Repository
```
git clone https://github.com/parthu-333/FlaskGoldPredict.git
cd FlaskGoldPredict
```
2. **Install Dependencies**
```
pip install
```
3. **Files Arrangement**
```
FlaskGoldPredict/
│── static/             # Static files (CSS, JavaScript, images)
│   │── Gold7.jpg       # Sample gold image
│   │── script.js       # JavaScript file for frontend functionality
│   │── styles.css      # CSS file for styling
│── templates/          # HTML templates
│   │── index.html      # Main frontend page
│── polynomial_regressin_model.pkl  # Machine learning model file
│── app.py              # Flask application
│── README.md           # Project documentation
```
And you should run the machine learning file in Google Colab to get .pkl file
```
Gold_vs_Year_(Polynomoial Regression).ipynb
```
4. **Run the Flask App**
```
python app.py
```
5. **Access the Web Application** Open your browser and go to:
```
http://127.0.0.1:5000/
```
## API Endpoints

| Endpoint  | Method | Description |
|-----------|--------|-------------|
| `/`       | GET    | Home page |
| `/predict`| POST   | Predicts gold price based on input data |
## Future Improvements
- Improve the machine learning model accuracy
- Add more economic indicators as features
- Deploy on a cloud platform
- Improve UI with better visualization

**Feel free to contribute and improve the project!**
