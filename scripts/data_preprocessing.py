
# Data Preprocessing Script
import pandas as pd
import numpy as np

# Load dataset
data_path = '/content/drive/My Drive/ADAS-Engineering-Lab/Sensor_Data_Analysis_Repo/data/signal_extractions/preprocessed_CAN_data.csv'
df = pd.read_csv(data_path)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Normalize signals
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
signal_features = [col for col in df.columns if "Signal" in col]
df[signal_features] = scaler.fit_transform(df[signal_features])

# Save preprocessed data
preprocessed_path = '/content/drive/My Drive/ADAS-Engineering-Lab/Sensor_Data_Analysis_Repo/data/signal_extractions/preprocessed_CAN_data.csv'
df.to_csv(preprocessed_path, index=False)
print(f"✅ Preprocessed dataset saved to {preprocessed_path}")
