# Stock Price Prediction using LSTM

This projgram creates a neural network to predict the change of an individual stock using historical stock price data and relevant features.

## Features

1. **Data Fetching**: 
    - Uses `yfinance` to fetch historical stock data.
    - Calculates technical indicators like Simple Moving Averages (SMA) and Bollinger Bands.

2. **Data Preprocessing**:
    - Cleans and prepares the data, including handling missing values.
    - Scales the data using `MinMaxScaler`.
    - Splits data into training, validation, and testing sets.

3. **Neural Network**:
    - Implements an LSTM model using TensorFlow/Keras.
    - Optional Attention mechanism for improved performance.

4. **Model Evaluation**:
    - Evaluates model performance using metrics like MAE, RMSE, and R-squared.
    - Plots predictions versus actual prices on the test set.
    - Plots learning curves during training.

5. **Hyperparameter Optimization**:
    - Uses Optuna for hyperparameter tuning to find the best model configuration.

## Requirements

- Python 3.x
- pandas
- numpy
- yfinance
- scikit-learn
- tensorflow
- matplotlib
- optuna

## Installation

Install the required packages using pip:

```bash
pip install pandas numpy yfinance scikit-learn tensorflow matplotlib optuna
```

## Usage

Run the script:

```bash
python marketpredictor.py
```

## Code Structure

- `fetch_stock_data(ticker, start_date, end_date)`: Fetches stock data and calculates technical indicators.
- `preprocess_data(data, feature_columns, target_column, lookback)`: Preprocesses the data for model training.
- `build_lstm_model(input_shape, use_attention=False)`: Builds the LSTM model with optional Attention mechanism.
- `evaluate_model(model, X_test, y_test, scaler_target)`: Evaluates the model performance.
- `plot_results(y_true, y_pred, filename)`: Plots the predictions versus actual prices.
- `plot_training_data(history, filename)`: Plots the learning curves during training.
- `optimize_hyperparameters(X_train, y_train, X_val, y_val, input_shape)`: Optimizes hyperparameters using Optuna.

## Suggestions for Improvement

- Experiment with different hyperparameters and model architectures.
- Incorporate additional technical indicators or features.
- Use more advanced preprocessing techniques to handle missing values and outliers.