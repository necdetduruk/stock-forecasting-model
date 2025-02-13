# Stock Forecasting Model

A robust stock forecasting model built using ensemble machine learning techniques (Random Forest and XGBoost) with dynamic prediction intervals, business day handling, drift adjustment, and robust logging. This project forecasts stock prices based on historical data and technical indicators, making it a valuable tool for financial analysis and decision-making.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
    - [Data Collection](#data-collection)
    - [Model Training](#model-training)
    - [Forecasting](#forecasting)
- [Model Maintenance](#model-maintenance)
- [Results and Visualizations](#results-and-visualizations)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## Overview <a name="overview"></a>
This project implements a stock forecasting model that uses historical stock data and technical indicators to predict future prices. The model leverages ensemble methods (combining Random Forest and XGBoost) along with a drift adjustment and dynamic prediction intervals. It also handles business days (skipping weekends and U.S. market holidays) and logs progress at every step, making it a practical tool for professional financial forecasting.

## Features <a name="features"></a>
- **Ensemble Forecasting:** Combines predictions from Random Forest and XGBoost models.
- **Dynamic Prediction Intervals:** Computes 95% prediction intervals using a rolling standard deviation of recent residuals.
- **Business Day Handling:** Forecasts are generated only for business days, skipping weekends and U.S. holidays via a custom business day offset.
- **Drift Adjustment:** Incorporates a drift term based on historical average daily returns to avoid flat forecasts.
- **Robust Logging and Status Updates:** Detailed console and log file outputs for monitoring the forecasting process.
- **Forecast Export:** Saves forecast results as a CSV file and generates visualization plots.

## Project Structure <a name="project-structure"></a>
stock-forecasting-model/ ├── data/ # (Optional) Folder for data files ├── notebooks/ # Jupyter notebooks demonstrating project usage ├── src/ # Source code for training and forecasting │ ├── train.py # (Optional) Training script for model development │ └── forecast.py # Forecasting script with business day handling, drift adjustment, and dynamic intervals ├── requirements.txt # List of dependencies ├── README.md # This file └── LICENSE # License file (if applicable)

## Installation
**Clone the Repository:**

```bash
   git clone https://github.com/your-username/stock-forecasting-model.git
   cd stock-forecasting-model
   ```

**Create a Virtual Environment (recommended):**

```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

**Install Dependencies**

```bash
    pip install -r requirements.txt
   ```

## Usage

**Data Collection**

<li>Ensure you have your historical stock data saved as {STOCK_TICKER}_historical_data.csv.</li>
<li>You may use a separate data collection script (e.g., using yfinance) to fetch and save the data if needed.</li>

**Model Training**
<li> Run the training script (if you need to retrain the model) to produce a trained model saved as {STOCK_TICKER}_best_model.pkl.</li>
<li> Example:

```bash
    python src/train.py
   ```
</li>
<li>This script uses hyperparameter optimization (RandomizedSearchCV and Optuna) to find the best model.</li>

**Forecasting**
<li>Run the forecasting script:</li>

    python src/forecast.py

<li>You will be prompted to enter:</li>

    1. The stock ticker (e.g., GME)
    2. The forecast horizon (number of business days)

<li>The script will:</li>

    1. Load the best model
    2. Load historical data
    3. Compute past in-sample predictions
    4. Generate an iterative forecast (handling business days, dynamic prediction intervals, and drift adjustment)
    5. Save the forecast as {STOCK_TICKER}_{FORECAST_DAYS}_forecast.csv
    6. Generate and display a forecast plot with actual historical prices, past predictions, future forecasts, and prediction intervals

**Model Maintenance**

    1. Monitoring: Set up automated monitoring of key performance metrics (MSE, MAE, R²) and trigger alerts if performance degrades.
    2. Retraining: Periodically retrain the model with the latest data (e.g., weekly, monthly, or quarterly) to adapt to changing market conditions.
    3. Data Drift: Continuously monitor for changes in the statistical properties of your input data.
    4. Feedback Loop: Use stakeholder feedback and post-mortem analyses after significant market events to refine your model and its features.
    5. Versioning: Maintain version control of your data, code, and model artifacts using Git and possibly model versioning tools.

**Results and Visualizations**

    The forecasting script produces a plot that shows:
        1. Historical actual close prices (blue)
        2. In-sample past predictions (orange)
        3. Future forecast values (red dashed) with 95% prediction intervals (red shaded)
    The forecast data is also saved as a CSV file for further analysis.

**Future Improvements**

    Forecast Returns Instead of Prices: Train the model to predict daily returns and accumulate them to generate more dynamic forecasts.
    Exogenous Variables: Incorporate external data (e.g., sentiment scores, macroeconomic indicators) to improve forecast accuracy.
    Advanced Models: Explore alternative time-series models such as LSTM, Prophet, or ARIMA.
    Enhanced Prediction Intervals: Use bootstrapping or Bayesian techniques for more adaptive prediction intervals.
    Deployment: Develop a RESTful API (using Flask or FastAPI) to enable real-time forecasting.
    Dashboard Integration: Create interactive dashboards (using Streamlit, Dash, or Tableau) for visualization and analysis.

**Contributing**

    Contributions are welcome! Please fork the repository and submit a pull request. Ensure that your changes are well-documented and adhere to the project's coding style guidelines.

**License**

    This project is licensed under the MIT License.
