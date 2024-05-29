import pandas as pd
from prophet import Prophet
import pickle

# Load your data
data = pd.read_csv('data.csv')
data.rename(columns={'date': 'ds', 'value': 'y'}, inplace=True)

# Train the model
model = Prophet()
model.fit(data)

# Save the model
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)
