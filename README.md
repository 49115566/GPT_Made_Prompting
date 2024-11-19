# Stock Price Prediction using LSTM

This project demonstrates how to develop a Python program that creates a neural network to predict the price of an individual stock using historical stock price data and relevant features.
It was made with a prompt telling ChatGPT to make a prompt, and the code is the result!

## Features

1. **Data Fetching**: 
    - Uses `yfinance` to fetch historical stock data.
    - Calculates technical indicators like RSI and MACD.

2. **Data Preprocessing**:
    - Cleans and prepares the data, including handling missing values.
    - Scales the data using `MinMaxScaler`.
    - Splits data into training, validation, and testing sets.

3. **Neural Network**:
    - Implements an LSTM model using TensorFlow/Keras.
    - Configurable layers for time series forecasting.

4. **Model Evaluation**:
    - Evaluates model performance using metrics like MAE, RMSE, and R-squared.
    - Plots predictions versus actual prices on the test set.

## Requirements

- Python 3.x
- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- matplotlib

## Installation

Install the required packages using pip:

```bash
pip install pandas numpy yfinance scikit-learn tensorflow matplotlib
```

## Usage

Run the script:

```bash
python marketpredictor.py
```

## Code Structure

- `fetch_stock_data(ticker, start_date, end_date)`: Fetches stock data and calculates technical indicators.
- `calculate_rsi(series, period)`: Calculates the Relative Strength Index (RSI).
- `calculate_macd(series, fast_period, slow_period, signal_period)`: Calculates the Moving Average Convergence Divergence (MACD).
- `preprocess_data(data, feature_columns, target_column, lookback)`: Preprocesses the data for model training.
- `build_lstm_model(input_shape)`: Builds the LSTM model.
- `evaluate_model(model, X_test, y_test, scaler, target_column)`: Evaluates the model performance.
- `plot_predictions(y_test, predictions, title)`: Plots the predictions versus actual prices.

## Suggestions for Improvement

- Experiment with different hyperparameters and model architectures.
- Incorporate additional technical indicators or features.
- Use more advanced preprocessing techniques to handle missing values and outliers.