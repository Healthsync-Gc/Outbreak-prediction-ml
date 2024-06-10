from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from prophet import Prophet
import os
import pickle



app = Flask(__name__)
CORS(app)

def load_model(region):
    model_path = f'models/{region}_model.pkl'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found for region: {region}")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def calculate_resource_needs(prediction_data, standard_requirements):
    total_resources_needed = {key: 0 for key in standard_requirements.keys()}
    for index, row in prediction_data.iterrows():
        yhat = row['yhat']
        for resource, multiplier in standard_requirements.items():
            total_resources_needed[resource] += int(multiplier * yhat)

    return total_resources_needed

def compare_with_available_resources(predicted_needs, available_resources):
    recommendations = {}
    for resource, needed in predicted_needs.items():
        available = available_resources.get(resource, 0)
        if needed > available:
            recommendations[resource] = needed - available
    return recommendations

@app.route('/predict', methods=['POST'])
def predict():
    # print(request)
    content = request.json
    region = content['region']
    start_date = content['start_date']
    end_date = content['end_date']
    available_resources = content['available_resources']
    standard_requirements = content['standard_requirements']
    print(standard_requirements)
    # return content
    try:
        model = load_model(region)

        # Generate future dates within the specified range
        future_dates = pd.date_range(start=start_date, end=end_date).to_frame(index=False, name='ds')
        
        # Make predictions
        forecast = model.predict(future_dates)

        
        # Extract the relevant predictions
        predictions = forecast[['ds', 'yhat']]
        # print(type(predictions))
        # return predictions
        # Get resource recommendations
        needed_resource = calculate_resource_needs(predictions,standard_requirements)

        recommendations = compare_with_available_resources(needed_resource,available_resources)
        # Convert predictions to a list of dictionaries for JSON serialization
        predictions_list = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        response = {
            "prediction":predictions_list,
            "recomendation":[{"name" : key, "value": val} for key, val in recommendations.items()]
            }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

if __name__ == '__main__':
    app.run("127.0.0.1", port=5000, debug=True)
