from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import pickle
from datetime import datetime
from prophet import Prophet
import io

app = Flask(__name__)

# Function to load model based on region
def load_model(region):
    model_path = f'models/{region}_model.pkl'
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json
    region = user_input['region']
    periods = int(user_input['periods'])
    frequency = user_input['frequency']

    # Load the appropriate model based on the selected region
    model = load_model(region)

    # Generate future dates
    future_dates = pd.date_range(start=datetime.now(), periods=periods, freq=frequency).to_frame(index=False, name='ds')
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    # Return the forecast as JSON
    return jsonify(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))

@app.route('/predict_table', methods=['POST'])
def predict_table():
    region = request.form['region']
    periods = int(request.form['periods'])
    frequency = request.form['frequency']

    # Load the appropriate model based on the selected region
    model = load_model(region)

    # Generate future dates
    future_dates = pd.date_range(start=datetime.now(), periods=periods, freq=frequency).to_frame(index=False, name='ds')
    
    # Make predictions
    forecast = model.predict(future_dates)
    
    # Convert forecast to JSON-like structure
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
    
    return render_template('results.html', forecast=forecast_data)

@app.route('/download_csv', methods=['POST'])
def download_csv():
    region = request.form['region']
    periods = int(request.form['periods'])
    frequency = request.form['frequency']

    # Load the appropriate model based on the selected region
    model = load_model(region)

    # Generate future dates
    future_dates = pd.date_range(start=datetime.now(), periods=periods, freq=frequency).to_frame(index=False, name='ds')
    
    # Make predictions
    forecast = model.predict(future_dates)

    # Convert forecast to CSV
    forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)

    # Create a bytes buffer to hold the CSV data
    buffer = io.BytesIO()
    buffer.write(forecast_csv.encode('utf-8'))
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name='forecast.csv', mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)
