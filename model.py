from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(X, y, epochs=5, batch_size=32):
    model = create_lstm_model((X.shape[1], 1))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_future(model, recent_seq, scaler):
    pred_scaled = model.predict(recent_seq)
    pred = scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])
