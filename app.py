from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import pickle
from datetime import datetime
from prophet import Prophet
import io
import os
from flask_cors import CORS
from urllib.parse import quote as url_quote

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Function to load model based on region
def load_model(region):
    model_path = f'models/{region}_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for region: {region}")
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
    start_date = user_input['start_date']
    end_date = user_input['end_date']

    try:
        # Load the appropriate model based on the selected region
        model = load_model(region)

        # Generate future dates within the specified range
        future_dates = pd.date_range(start=start_date, end=end_date).to_frame(index=False, name='ds')
        
        # Make predictions
        forecast = model.predict(future_dates)
        
        # Return the forecast as JSON
        return jsonify(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_table', methods=['POST'])
def predict_table():
    region = request.form['region']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    try:
        # Load the appropriate model based on the selected region
        model = load_model(region)

        # Generate future dates within the specified range
        future_dates = pd.date_range(start=start_date, end=end_date).to_frame(index=False, name='ds')
        
        # Make predictions
        forecast = model.predict(future_dates)
        
        # Convert forecast to JSON-like structure
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        
        return render_template('results.html', forecast=forecast_data)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/download_csv', methods=['POST'])
def download_csv():
    region = request.form['region']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    try:
        # Load the appropriate model based on the selected region
        model = load_model(region)

        # Generate future dates within the specified range
        future_dates = pd.date_range(start=start_date, end=end_date).to_frame(index=False, name='ds')
        
        # Make predictions
        forecast = model.predict(future_dates)

        # Convert forecast to CSV
        forecast_csv = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)

        # Create a bytes buffer to hold the CSV data
        buffer = io.BytesIO()
        buffer.write(forecast_csv.encode('utf-8'))
        buffer.seek(0)

        return send_file(buffer, as_attachment=True, download_name='forecast.csv', mimetype='text/csv')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
