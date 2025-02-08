
# Anomaly Detection & Machine Learning Models Script
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

# Load dataset
data_path = '/content/drive/My Drive/ADAS-Engineering-Lab/Sensor_Data_Analysis_Repo/data/signal_extractions/preprocessed_CAN_data.csv'
df = pd.read_csv(data_path)

# Prepare data
signal_features = [col for col in df.columns if "Signal" in col]
X = df[signal_features].fillna(df[signal_features].mean())
y = df["Label"]

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Random Forest Model
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
print(classification_report(y_test, rf_preds))

# LSTM Model
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
y_lstm = y
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)
lstm_model = Sequential([
    LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dense(1, activation="sigmoid")
])
lstm_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=5, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))
print("✅ LSTM Model Trained")
