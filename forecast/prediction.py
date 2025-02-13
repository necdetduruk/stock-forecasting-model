import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import pickle
import time
from datetime import timedelta
from pandas.tseries.offsets import CustomBusinessDay
import holidays
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler

# Configure logging
logging.basicConfig(
    filename="forecast_prediction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# List of feature columns used during training
FEATURE_COLUMNS = ['SMA_10', 'SMA_50', 'Momentum', 'Volatility',
                   'Lag_1', 'Month', 'DayOfWeek', 'RSI', 'BB_upper', 'BB_lower']

# Define a custom business day offset that skips weekends and U.S. market holidays.
us_holidays = holidays.US(years=range(2000, 2030))
custom_bd = CustomBusinessDay(holidays=list(us_holidays))

class ForecastPredictor:
    def __init__(self, stock_ticker, forecast_days):
        self.stock_ticker = stock_ticker.upper()
        self.forecast_days = forecast_days  # forecast horizon in business days
        self.best_model = None  # Dictionary loaded from pickle
        self.historical_data = None  # DataFrame loaded from CSV
        self.forecast_df = None  # DataFrame after iterative forecasting

    def load_best_model(self):
        """Load the saved best model from a pickle file."""
        model_filename = f"{self.stock_ticker}_best_model.pkl"
        print(f"STATUS: Loading best model from '{model_filename}'...")
        if not os.path.exists(model_filename):
            raise FileNotFoundError(f"Best model file '{model_filename}' not found.")
        with open(model_filename, "rb") as f:
            self.best_model = pickle.load(f)
        logging.info(f"Loaded best model from {model_filename}.")
        print("STATUS: Best model loaded successfully.")

    def load_historical_data(self):
        """Load the historical data CSV for the stock."""
        data_filename = f"{self.stock_ticker}_historical_data.csv"
        print(f"STATUS: Loading historical data from '{data_filename}'...")
        if not os.path.exists(data_filename):
            raise FileNotFoundError(f"Historical data file '{data_filename}' not found.")
        df = pd.read_csv(data_filename, parse_dates=['Date'], index_col='Date')
        # Ensure the index is timezone-naive
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)
        self.historical_data = df
        logging.info(f"Loaded historical data with {len(df)} records.")
        print("STATUS: Historical data loaded successfully.")

    @staticmethod
    def compute_features(df):
        """
        Compute technical indicators to match training.
        Expected features:
          - SMA_10, SMA_50, Momentum, Volatility, Lag_1, Month, DayOfWeek, RSI, BB_upper, BB_lower
        """
        df_feat = df.copy()
        df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
        df_feat['SMA_50'] = df_feat['Close'].rolling(window=50).mean()
        df_feat['Momentum'] = df_feat['Close'] - df_feat['Close'].shift(5)
        df_feat['Volatility'] = df_feat['Close'].rolling(window=10).std()
        df_feat['Lag_1'] = df_feat['Close'].shift(1)
        df_feat['Month'] = df_feat.index.month
        df_feat['DayOfWeek'] = df_feat.index.dayofweek

        # RSI (Relative Strength Index) with a 14-day window
        delta = df_feat['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df_feat['RSI'] = 100 - (100 / (1 + rs))

        # Bollinger Bands (20-day moving average ± 2*std)
        df_feat['MA_20'] = df_feat['Close'].rolling(window=20).mean()
        rolling_std = df_feat['Close'].rolling(window=20).std()
        df_feat['BB_upper'] = df_feat['MA_20'] + 2 * rolling_std
        df_feat['BB_lower'] = df_feat['MA_20'] - 2 * rolling_std

        # Keep only the features we need
        df_feat = df_feat[FEATURE_COLUMNS]
        return df_feat

    def compute_dynamic_error_std(self, window=30):
        """
        Compute the dynamic error standard deviation from the most recent 'window' days
        of historical residuals (actual Close minus in-sample predictions).
        """
        past_preds = self.compute_past_predictions()
        common_idx = self.historical_data.index.intersection(past_preds.index)
        residuals = self.historical_data.loc[common_idx, 'Close'] - past_preds.loc[common_idx]
        if len(residuals) >= window:
            recent_resid = residuals[-window:]
        else:
            recent_resid = residuals
        dynamic_std = np.std(recent_resid)
        return dynamic_std

    def forecast(self):
        """
        Perform an iterative forecast for the specified number of business days.
        For each forecast day:
          - Add the next business day (using a custom business day offset that accounts for holidays),
          - Recompute features,
          - Scale the new feature row (as a DataFrame with proper column names),
          - Use the ensemble (weighted RF and XGB) to predict the Close,
          - Compute a 95% prediction interval using a dynamic error std (rolling window of residuals),
          - Apply a drift adjustment to overcome flat predictions.
        """
        print("STATUS: Starting iterative forecast...")
        df_forecast = self.historical_data.copy()
        # Prepare new columns for forecast intervals
        df_forecast['Forecast_lower'] = np.nan
        df_forecast['Forecast_upper'] = np.nan

        forecast_values = []  # List to store (date, predicted_close, lower_bound, upper_bound)

        # Precompute drift adjustment based on historical daily returns.
        historical_returns = self.historical_data['Close'].pct_change().dropna()
        drift = historical_returns.mean()  # average daily return
        # Use a fraction (e.g., half) of the drift for adjustment
        drift_adjustment = 0.5 * drift
        print(f"STATUS: Computed drift adjustment factor: {drift_adjustment:.4f}")

        for i in range(self.forecast_days):
            print(f"STATUS: Forecasting business day {i+1} of {self.forecast_days}...")
            last_date = df_forecast.index[-1]
            # Add one custom business day (skipping weekends and US holidays)
            next_date = last_date + custom_bd
            # Append a new row with NaN for 'Close'
            new_row = pd.DataFrame({'Close': [np.nan]}, index=[next_date])
            df_forecast = pd.concat([df_forecast, new_row])
            # Recompute features on the entire DataFrame
            df_features = self.compute_features(df_forecast)
            # Check if features for the new date are available
            if next_date not in df_features.index:
                predicted_close = df_forecast['Close'].iloc[-2]
                print(f"WARNING: Features not available for {next_date}; using previous close as prediction.")
            else:
                last_features = df_features.loc[next_date]
                if last_features.isnull().any():
                    predicted_close = df_forecast['Close'].iloc[-2]
                    print(f"WARNING: Some features are NaN for {next_date}; using previous close as prediction.")
                else:
                    # Convert the new feature row to a DataFrame with proper column names
                    X_new_df = pd.DataFrame(last_features.values.reshape(1, -1), columns=FEATURE_COLUMNS)
                    X_new_scaled = self.best_model['scaler'].transform(X_new_df)
                    # Ensemble prediction: weighted average of RF and XGB predictions
                    pred_rf = self.best_model['rf_model'].predict(X_new_scaled)[0]
                    pred_xgb = self.best_model['xgb_model'].predict(X_new_scaled)[0]
                    predicted_close = self.best_model['w_rf'] * pred_rf + self.best_model['w_xgb'] * pred_xgb
            # Apply drift adjustment so forecast does not remain flat
            predicted_close = predicted_close * (1 + drift_adjustment)

            # Dynamically compute error std (from the most recent 30 days of residuals)
            current_error_std = self.compute_dynamic_error_std(window=30)
            print(f"STATUS: Dynamic error std for prediction interval: {current_error_std:.4f}")
            # Compute 95% prediction interval
            forecast_lower = predicted_close - 1.96 * current_error_std
            forecast_upper = predicted_close + 1.96 * current_error_std

            # Update the new row's 'Close' and interval columns with the predicted values
            df_forecast.at[next_date, 'Close'] = predicted_close
            df_forecast.at[next_date, 'Forecast_lower'] = forecast_lower
            df_forecast.at[next_date, 'Forecast_upper'] = forecast_upper

            forecast_values.append((next_date, predicted_close, forecast_lower, forecast_upper))
            print(f"STATUS: {next_date.date()} forecasted as {predicted_close:.2f} " +
                  f"(95% PI: [{forecast_lower:.2f}, {forecast_upper:.2f}]).")
        self.forecast_df = df_forecast
        print("STATUS: Iterative forecast complete.")
        return forecast_values

    def compute_past_predictions(self):
        """
        Compute in-sample predictions on the historical data using the best model.
        Only use rows where all required features are available.
        """
        print("STATUS: Computing past in-sample predictions...")
        df_features = self.compute_features(self.historical_data)
        predictions = pd.Series(index=df_features.index, dtype=float)
        for idx in df_features.index:
            features = df_features.loc[idx]
            X_df = pd.DataFrame(features.values.reshape(1, -1), columns=FEATURE_COLUMNS)
            X_scaled = self.best_model['scaler'].transform(X_df)
            pred_rf = self.best_model['rf_model'].predict(X_scaled)[0]
            pred_xgb = self.best_model['xgb_model'].predict(X_scaled)[0]
            pred = self.best_model['w_rf'] * pred_rf + self.best_model['w_xgb'] * pred_xgb
            predictions.loc[idx] = pred
        print("STATUS: Past predictions computed.")
        return predictions

    def plot_forecast(self, past_predictions, forecast_start_date):
        """
        Plot:
          - Historical actual Close prices,
          - In-sample past predictions, and
          - Future forecast for the specified horizon along with prediction intervals.
        A vertical line is drawn at the forecast start date.
        """
        print("STATUS: Plotting forecast results...")
        plt.figure(figsize=(12, 6))
        # Plot historical actual data
        plt.plot(self.historical_data.index, self.historical_data['Close'],
                 label="Actual Historical", color='blue')
        # Plot in-sample past predictions
        plt.plot(past_predictions.index, past_predictions,
                 label="Past Predictions", color='orange')
        # Plot future forecast (rows after the last historical date)
        forecast_data = self.forecast_df[self.forecast_df.index > forecast_start_date]
        plt.plot(forecast_data.index, forecast_data['Close'],
                 label="Future forecast", linestyle='--', color='red')
        # Plot prediction intervals for future forecast
        plt.fill_between(forecast_data.index,
                         forecast_data['Forecast_lower'],
                         forecast_data['Forecast_upper'],
                         color='red', alpha=0.3,
                         label='95% Prediction Interval')
        # Draw a vertical line at the forecast start
        plt.axvline(x=forecast_start_date, color='green', linestyle=':', label="forecast Start")
        plt.xlabel("Date")
        plt.ylabel("Stock Close Price")
        plt.title(f"{self.stock_ticker} forecast")
        plt.legend()
        plot_filename = f"{self.stock_ticker}_forecast.png"
        plt.savefig(plot_filename)
        plt.show()
        logging.info(f"forecast plot saved as {plot_filename}")
        print(f"STATUS: forecast plot saved as '{plot_filename}'.")

def main():
    try:
        print("STATUS: Forecasting process started.")
        stock_ticker = input("Enter stock ticker: ").strip().upper()
        forecast_days = int(input("Enter forecast horizon in business days: ").strip())
        predictor = ForecastPredictor(stock_ticker, forecast_days)
        predictor.load_best_model()
        predictor.load_historical_data()
        past_predictions = predictor.compute_past_predictions()
        # forecast start is the last date of the historical data
        forecast_start_date = predictor.historical_data.index[-1]
        predictor.forecast()  # Perform the iterative forecast
        # Save forecast DataFrame as CSV
        forecast_filename = f"{predictor.stock_ticker}_{predictor.forecast_days}_forecast.csv"
        predictor.forecast_df.to_csv(forecast_filename)
        print(f"STATUS: forecast data saved as '{forecast_filename}'.")
        logging.info(f"forecast data saved as '{forecast_filename}'.")
        predictor.plot_forecast(past_predictions, forecast_start_date)
        print("STATUS: forecast completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
