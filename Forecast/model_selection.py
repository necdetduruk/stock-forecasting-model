import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import optuna  # Bayesian Optimization
import time
import pickle
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler  # Robust scaling for outlier-prone data

# Configure logging
logging.basicConfig(
    filename="stock_forecasting.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class HybridStockPredictor:
    def __init__(self, stock_ticker, horizon):
        self.stock_ticker = stock_ticker.upper()
        self.horizon = horizon
        self.data = None
        self.rf_model = None
        self.xgb_model = None
        self.remove_outliers = True  # Set to True to remove outliers from training data

    def load_data(self):
        """Loads historical stock data from a CSV file."""
        try:
            print(f"📥 Loading data for {self.stock_ticker}...")
            file_path = f"{self.stock_ticker}_historical_data.csv"
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"⚠️ File '{file_path}' not found. Please provide valid stock data.")

            # Read the CSV with 'Date' parsed as dates and set as index
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            # Convert the index to UTC and then remove timezone info to make it timezone-naive
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(None)

            df['Return'] = df['Close'].pct_change()
            df.dropna(inplace=True)

            # Filter based on user-defined horizon (in years)
            df = df.loc[df.index >= df.index.max() - pd.DateOffset(years=self.horizon)]
            self.data = df
            print(f"✅ Loaded {len(df)} records for {self.stock_ticker}.\n")
            logging.info(f"Loaded {len(df)} records for {self.stock_ticker}.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def add_technical_indicators(self):
        """Adds technical indicators, including new features RSI and Bollinger Bands."""
        try:
            print("📊 Adding technical indicators...")
            # Existing indicators
            self.data['SMA_10'] = self.data['Close'].rolling(window=10).mean()
            self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
            self.data['Momentum'] = self.data['Close'] - self.data['Close'].shift(5)
            self.data['Volatility'] = self.data['Close'].rolling(window=10).std()
            self.data['Lag_1'] = self.data['Close'].shift(1)
            self.data['Month'] = self.data.index.month
            self.data['DayOfWeek'] = self.data.index.dayofweek

            # New feature: RSI (Relative Strength Index)
            delta = self.data['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = avg_gain / avg_loss
            self.data['RSI'] = 100 - (100 / (1 + rs))

            # New features: Bollinger Bands
            self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
            self.data['BB_upper'] = self.data['MA_20'] + 2 * self.data['Close'].rolling(window=20).std()
            self.data['BB_lower'] = self.data['MA_20'] - 2 * self.data['Close'].rolling(window=20).std()

            self.data.dropna(inplace=True)
            print("✅ Technical indicators added.\n")
            logging.info("Technical indicators added with RSI and Bollinger Bands.")
        except Exception as e:
            logging.error(f"Error generating indicators: {e}")
            raise

    def prepare_data(self):
        """Splits data into features and target."""
        try:
            print("📆 Preparing data...")
            self.add_technical_indicators()
            # Updated feature list including new indicators
            features = ['SMA_10', 'SMA_50', 'Momentum', 'Volatility',
                        'Lag_1', 'Month', 'DayOfWeek', 'RSI', 'BB_upper', 'BB_lower']
            X = self.data[features]
            y = self.data['Close']
            return X, y
        except Exception as e:
            logging.error(f"Error preparing data: {e}")
            raise

    def remove_outliers_fn(self, X, y, threshold=3):
        """Removes outliers from training data based on the target variable."""
        mean_y = y.mean()
        std_y = y.std()
        mask = (np.abs(y - mean_y) <= threshold * std_y)
        return X[mask], y[mask]

    def train_rf_model(self, X_train, y_train):
        """Uses Randomized Search to optimize Random Forest."""
        try:
            print("🔍 Running Randomized Search for Random Forest...")
            rf = RandomForestRegressor(n_jobs=-1, random_state=42)
            param_dist = {
                'n_estimators': [50, 100, 200, 300, 400],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4, 6]
            }
            # Increase the number of iterations to explore a wider parameter space
            random_search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=20,
                cv=3,
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(X_train, y_train)
            self.rf_model = random_search.best_estimator_
            print(f"✅ Best RF Params: {random_search.best_params_}\n")
            logging.info(f"Best RF Params: {random_search.best_params_}")
        except Exception as e:
            logging.error(f"Error training Random Forest: {e}")
            raise

    def train_xgb_model(self, X_train, y_train, X_valid, y_valid):
        """Uses Bayesian Optimization to optimize XGBoost."""
        try:
            print("🧠 Running Bayesian Optimization for XGBoost...")
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 400),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
                model = XGBRegressor(**params, n_jobs=-1, random_state=42, early_stopping_rounds=10)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    verbose=False
                )
                preds = model.predict(X_valid)
                return mean_squared_error(y_valid, preds)

            # Increase the number of trials to further search the hyperparameter space
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            best_params = study.best_params
            print(f"✅ Best XGBoost Params: {best_params}\n")
            logging.info(f"Best XGBoost Params: {best_params}")
            self.xgb_model = XGBRegressor(**best_params, n_jobs=-1, random_state=42)
            self.xgb_model.fit(X_train, y_train)
        except Exception as e:
            logging.error(f"Error training XGBoost: {e}")
            raise

    def run(self):
        """Executes the full pipeline with cross-validation, robust scaling, weighted ensemble,
        plots results with confidence intervals, and saves the best model."""
        try:
            start_time = time.time()
            self.load_data()
            X, y = self.prepare_data()

            # Use TimeSeriesSplit (an expanding window approach) for cross-validation.
            tscv = TimeSeriesSplit(n_splits=5)
            mse_list = []
            r2_list = []
            fold = 1
            last_X_test = None
            last_y_test = None
            last_hybrid_preds = None

            # Variables to track the best fold's performance and models
            best_mse = np.inf
            best_model_dict = None

            print("🔄 Starting Time Series Cross-Validation...")
            for train_index, test_index in tscv.split(X):
                print(f"----- Fold {fold} -----")
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Log target statistics for the fold
                print(f"Fold {fold} - y_train: mean={y_train.mean():.2f}, std={y_train.std():.2f}; "
                      f"y_test: mean={y_test.mean():.2f}, std={y_test.std():.2f}")
                logging.info(f"Fold {fold} - y_train: mean={y_train.mean():.2f}, std={y_train.std():.2f}; "
                             f"y_test: mean={y_test.mean():.2f}, std={y_test.std():.2f}")

                # Optionally remove outliers from training data
                if self.remove_outliers:
                    X_train, y_train = self.remove_outliers_fn(X_train, y_train)
                    print(f"Fold {fold} - After outlier removal: y_train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")
                    logging.info(f"Fold {fold} - After outlier removal: y_train: mean={y_train.mean():.2f}, std={y_train.std():.2f}")

                # Scale features using RobustScaler
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train both models on the current fold using scaled features
                self.train_rf_model(X_train_scaled, y_train)
                self.train_xgb_model(X_train_scaled, y_train, X_test_scaled, y_test)

                # Generate predictions from both models
                preds_rf = self.rf_model.predict(X_test_scaled)
                preds_xgb = self.xgb_model.predict(X_test_scaled)

                # Compute individual errors for the weighted ensemble
                epsilon = 1e-6
                error_rf = mean_squared_error(y_test, preds_rf)
                error_xgb = mean_squared_error(y_test, preds_xgb)
                w_rf = 1 / (error_rf + epsilon)
                w_xgb = 1 / (error_xgb + epsilon)
                total_w = w_rf + w_xgb
                w_rf_norm = w_rf / total_w
                w_xgb_norm = w_xgb / total_w
                print(f"Fold {fold} - Weights: RF: {w_rf_norm:.2f}, XGB: {w_xgb_norm:.2f}")
                logging.info(f"Fold {fold} - Weights: RF: {w_rf_norm:.2f}, XGB: {w_xgb_norm:.2f}")

                # Weighted ensemble prediction
                hybrid_preds = w_rf_norm * preds_rf + w_xgb_norm * preds_xgb

                # Evaluate the hybrid model
                mse = mean_squared_error(y_test, hybrid_preds)
                r2 = r2_score(y_test, hybrid_preds)
                mse_list.append(mse)
                r2_list.append(r2)
                print(f"Fold {fold} - MSE: {mse:.6f}, R²: {r2:.4f}\n")
                logging.info(f"Fold {fold} - MSE: {mse:.6f}, R²: {r2:.4f}")

                # If this fold has the best performance so far, save its models, weights, and scaler
                if mse < best_mse:
                    best_mse = mse
                    best_model_dict = {
                        "rf_model": self.rf_model,
                        "xgb_model": self.xgb_model,
                        "w_rf": w_rf_norm,
                        "w_xgb": w_xgb_norm,
                        "scaler": scaler
                    }
                # Save the last fold's data for plotting
                last_X_test = X_test
                last_y_test = y_test
                last_hybrid_preds = hybrid_preds
                fold += 1

            # Calculate average performance across folds
            avg_mse = np.mean(mse_list)
            avg_r2 = np.mean(r2_list)
            print(f"📊 Average MSE: {avg_mse:.6f}, Average R²: {avg_r2:.4f}")
            logging.info(f"Average MSE: {avg_mse:.6f}, Average R²: {avg_r2:.4f}")

            # Save the best model as a pickle file
            if best_model_dict is not None:
                best_model_filename = f"{self.stock_ticker}_best_model.pkl"
                with open(best_model_filename, "wb") as f:
                    pickle.dump(best_model_dict, f)
                print(f"💾 Best model saved as {best_model_filename} with fold MSE: {best_mse:.6f}")
                logging.info(f"Best model saved as {best_model_filename} with fold MSE: {best_mse:.6f}")

            # Plot predictions vs. actual for the last fold, with a confidence interval
            if last_y_test is not None:
                plt.figure(figsize=(10, 6))
                plt.plot(last_y_test.index, last_y_test, label="Actual", color='blue')
                plt.plot(last_y_test.index, last_hybrid_preds, label="Predicted", linestyle="--", color='orange')

                # Compute residuals and estimate a 95% confidence interval
                residuals = last_y_test.values - last_hybrid_preds
                error_std = np.std(residuals)
                # Assuming normality, the 95% CI is approximately ±1.96 standard deviations
                upper_bound = last_hybrid_preds + 1.96 * error_std
                lower_bound = last_hybrid_preds - 1.96 * error_std

                plt.fill_between(last_y_test.index, lower_bound, upper_bound, color='gray', alpha=0.3,
                                 label='95% Confidence Interval')

                plt.xlabel("Date")
                plt.ylabel("Stock Close Price")
                plt.title(f"{self.stock_ticker} Actual vs Predicted Prices")
                plt.legend()
                plot_filename = f"{self.stock_ticker}_predictions.png"
                plt.savefig(plot_filename)
                plt.close()
                print(f"📈 Prediction plot saved as {plot_filename}")
                logging.info(f"Prediction plot saved as {plot_filename}")

            print(f"⏳ Total Execution Time: {time.time() - start_time:.2f} seconds\n")
        except Exception as e:
            logging.error(f"Error during run: {e}")
            print(f"❌ An error occurred: {e}")

def main():
    """Main function for user input and running the prediction model."""
    try:
        stock_ticker = input("Enter stock ticker: ").upper()
        horizon = int(input("Enter horizon in years: "))
        predictor = HybridStockPredictor(stock_ticker, horizon)
        predictor.run()
    except Exception as e:
        print(f"❌ An error occurred in main: {e}")
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
