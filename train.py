import os
import pandas as pd
from prophet import Prophet
import pickle

def train_and_save_models(data_dir, regions, output_dir):
    for region in regions:
        # Construct file path
        file_path = os.path.join(data_dir, f'{region}.csv')
        
        # Check if the file exists
        if not os.path.exists(file_path):
            print(f"Data file for {region} not found at {file_path}. Skipping this region.")
            continue
        
        # Load data
        data = pd.read_csv(file_path)
        data.rename(columns={'Date': 'ds', 'Confirmed': 'y'}, inplace=True)
        
        # Train the model
        model = Prophet()
        model.fit(data)
        
        # Save the model
        output_path = os.path.join(output_dir, f'{region}_model.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model for {region} saved at {output_path}.")

# Usage
regions = ["addis_ababa", "amhara", "oromia", "tigray", "somali", "afar", "diredawa", "debub", "gambela", "b_gumuz", "harari"]
data_dir = 'datas'  # Directory where the datasets are stored
output_dir = 'models'  # Directory where the models will be saved

train_and_save_models(data_dir, regions, output_dir)
