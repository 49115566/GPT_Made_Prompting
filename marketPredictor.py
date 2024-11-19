import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Step 1: Fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['RSI'] = calculate_rsi(data['Close'])
    data['MACD'], data['Signal'] = calculate_macd(data['Close'])
    return data

# Step 2: Technical indicators (RSI and MACD)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = series.ewm(span=fast_period, adjust=False).mean()
    ema_slow = series.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

# Step 3: Preprocessing data
def preprocess_data(data, feature_columns, target_column, lookback=60):
    data = data.dropna()
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[feature_columns + [target_column]])
    
    X, y = [], []
    for i in range(lookback, len(data_scaled)):
        X.append(data_scaled[i-lookback:i, :-1])
        y.append(data_scaled[i, -1])
    
    X, y = np.array(X), np.array(y)
    return X, y, scaler

# Step 4: Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Step 5: Train and evaluate model
def evaluate_model(model, X_test, y_test, scaler, target_column):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((predictions.shape[0], len(target_column) - 1)), predictions))
    )[:, -1]
    y_test_rescaled = scaler.inverse_transform(
        np.hstack((np.zeros((y_test.shape[0], len(target_column) - 1)), y_test.reshape(-1, 1)))
    )[:, -1]
    
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    
    print(f"MAE: {mae}, RMSE: {rmse}, R^2: {r2}")
    return predictions_rescaled, y_test_rescaled

# Step 6: Plot results
def plot_predictions(y_test, predictions, title='Model Predictions vs Actual'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.show()

# Main function to tie it all together
if __name__ == '__main__':
    # Fetch and preprocess data
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    data = fetch_stock_data(ticker, start_date, end_date)

    feature_columns = ['Close', 'Volume', 'RSI', 'MACD', 'Signal']
    target_column = 'Close'
    
    X, y, scaler = preprocess_data(data, feature_columns, target_column)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Build and train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)

    # Evaluate and plot results
    predictions, y_test_rescaled = evaluate_model(model, X_test, y_test, scaler, feature_columns)
    plot_predictions(y_test_rescaled, predictions)
