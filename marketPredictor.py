import os
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping
import optuna

# Step 1: Fetch and preprocess stock data
def fetch_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data['PriceChange'] = data['Close'].diff()  # Simplified price change calculation
    data['SMA20'] = data['Close'].rolling(window=20).mean()  # 20-day Simple Moving Average
    data['SMA50'] = data['Close'].rolling(window=50).mean()  # 50-day Simple Moving Average
    data['Volatility'] = data['Close'].rolling(window=20).std()  # Volatility (standard deviation)
    data['UpperBB'] = data['SMA20'] + (2 * data['Volatility'])  # Bollinger Upper Band
    data['LowerBB'] = data['SMA20'] - (2 * data['Volatility'])  # Bollinger Lower Band
    return data.dropna()

# Step 2: Preprocessing with separate scaling
def preprocess_data(data, feature_columns, target_column, lookback=60):
    data = data.dropna()
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler(feature_range=(-1, 1))

    features_scaled = scaler_features.fit_transform(data[feature_columns])
    target_scaled = scaler_target.fit_transform(data[[target_column]])

    X, y = [], []
    for i in range(lookback, len(features_scaled)):
        X.append(features_scaled[i - lookback:i])
        y.append(target_scaled[i, 0])
    
    return np.array(X), np.array(y), scaler_features, scaler_target

# Step 3: Build LSTM with optional Attention mechanism
def build_lstm_model(input_shape, units, dropout, learning_rate, use_attention=False):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        BatchNormalization(),
        Dropout(dropout),
        LSTM(units, return_sequences=use_attention),
        BatchNormalization(),
        Dropout(dropout),
    ])
    if use_attention:
        model.add(Attention())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error')
    return model

# Step 4: Train and evaluate the model
def evaluate_model(model, X_test, y_test, scaler_target, future_steps=0):
    predictions = model.predict(X_test)
    predictions_rescaled = scaler_target.inverse_transform(predictions)
    y_test_rescaled = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    
    mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions_rescaled))
    r2 = r2_score(y_test_rescaled, predictions_rescaled)
    
    print(f"MAE: {mae}, RMSE: {rmse}, R^2: {r2}")

    future_predictions = None
    if future_steps > 0:
        future_predictions = predict_future(model, X_test[-1], future_steps, scaler_target)
    
    return predictions_rescaled, y_test_rescaled, future_predictions

def predict_future(model, X_test, future_steps, scaler_target):
    predictions = []
    current_input = X_test.copy()
    
    for _ in range(future_steps):
        prediction = model.predict(current_input.reshape(1, *current_input.shape))
        predictions.append(prediction)
        current_input = np.append(current_input[1:], prediction, axis=0)
    
    predictions = np.array(predictions).reshape(-1, 1)
    return scaler_target.inverse_transform(predictions)

# Step 5: Plot results and learning curves
def plot_results(y_true, y_pred, filename='results.png'):
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, label='True Values')
    plt.plot(y_pred, label='Predictions')
    plt.title('Stock Price Predictions')
    plt.legend()
    plt.savefig(filename)

def plot_training_data(history, filename='learning_curve.png'):
    # Plot learning curves
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.savefig(filename)

# Step 6: Hyperparameter optimization using Optuna
def optimize_hyperparameters(X_train, y_train, X_val, y_val, input_shape):
    def objective(trial):
        units = trial.suggest_int('units', 32, 128)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        batch_size = trial.suggest_int('batch_size', 16, 64)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2)

        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(dropout),
            LSTM(units, return_sequences=False),
            BatchNormalization(),
            Dropout(dropout),
            Dense(1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mean_squared_error')

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=20, batch_size=batch_size, verbose=0)
        return min(history.history['val_loss'])
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    return study.best_params

# Main function to tie everything together
if __name__ == '__main__':
    # Fetch data
    ticker = 'AAPL'
    start_date = '2015-01-01'
    end_date = '2023-12-31'
    data = fetch_stock_data(ticker, start_date, end_date)
    
    feature_columns = ['Close', 'Volume', 'SMA20', 'SMA50', 'Volatility', 'UpperBB', 'LowerBB']
    target_column = 'PriceChange'
    lookback = 60
    
    X, y, scaler_features, scaler_target = preprocess_data(data, feature_columns, target_column, lookback)
    
    # Train-test-validation split
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    
    model_path = 'best_model.keras'
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # Hyperparameter tuning
        input_shape = (X_train.shape[1], X_train.shape[2])
        best_params = optimize_hyperparameters(X_train, y_train, X_val, y_val, input_shape)

        # Train model with optimized parameters
        model = build_lstm_model(input_shape, best_params['units'], best_params['dropout'], best_params['learning_rate'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                            epochs=50, batch_size=best_params['batch_size'],
                            callbacks=[early_stopping])
        
        plot_training_data(history)
        model.save(model_path)

    # Evaluate and plot results
    predictions, y_test_rescaled, future_predictions = evaluate_model(model, X_test, y_test, scaler_target)
    plot_results(y_test_rescaled, predictions)
    if future_predictions is not None:
        print(f"Future predictions: {future_predictions}")
