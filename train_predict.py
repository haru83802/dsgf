import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

from config import TIME_STEP, MODEL_EPOCHS, MODEL_BATCH

def load_data(path):
    df = pd.read_csv(path)
    data = df[['Close']].values
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data, scaler

def prepare_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:i+time_step])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_and_predict(ticker, data_path):
    try:
        data, scaler = load_data(data_path)
        X, y = prepare_dataset(data, TIME_STEP)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = build_model((X.shape[1], 1))
        model.fit(X, y, epochs=MODEL_EPOCHS, batch_size=MODEL_BATCH, verbose=0)

        last_sequence = data[-TIME_STEP:].reshape(1, TIME_STEP, 1)
        predicted = model.predict(last_sequence)
        predicted_price = scaler.inverse_transform(predicted)[0][0]
        return predicted_price
    except Exception as e:
        print(f"{ticker} 예측 실패: {e}")
        return None
