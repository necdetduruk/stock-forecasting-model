import yfinance as yf
import pandas as pd
import logging
import os
from datetime import datetime
from pandas.tseries.offsets import DateOffset

# Configure logging
logging.basicConfig(
    filename="data_collection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class StockDataCollector:
    def __init__(self, stock_ticker, horizon):
        """
        Initialize the collector with a stock ticker and a horizon (in years).
        """
        self.stock_ticker = stock_ticker.upper()
        self.horizon = horizon  # Horizon in years
        self.data = None

    def collect_data(self):
        """
        Collect historical stock data-collection for the given ticker and horizon using yfinance.
        """
        try:
            logging.info(f"Starting data-collection collection for {self.stock_ticker} with a horizon of {self.horizon} years.")

            # Determine start and end dates
            end_date = pd.Timestamp.today()
            start_date = end_date - DateOffset(years=self.horizon)
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")
            logging.info(f"Collecting data-collection from {start_date_str} to {end_date_str} for {self.stock_ticker}.")

            # Download data-collection using yfinance
            data = yf.download(self.stock_ticker, start=start_date_str, end=end_date_str)
            if data.empty:
                error_msg = (f"No data-collection found for ticker {self.stock_ticker} "
                             f"between {start_date_str} and {end_date_str}.")
                logging.error(error_msg)
                raise ValueError(error_msg)

            # Check if the DataFrame has MultiIndex columns and flatten them if so.
            if isinstance(data.columns, pd.MultiIndex):
                # For example, if columns are like:
                #   Price     Close     High     Low      Open     Volume
                #   Ticker    GME       GME      GME      GME      GME
                # We'll drop the second level:
                data.columns = data.columns.get_level_values(0)
                logging.info("MultiIndex columns detected and flattened.")

            self.data = data
            logging.info(f"Successfully collected {len(data)} records for {self.stock_ticker}.")
        except Exception as e:
            logging.error(f"Error collecting data-collection for {self.stock_ticker}: {e}")
            raise

    def save_data(self):
        """
        Save the collected data-collection to a CSV file.
        """
        try:
            if self.data is None:
                raise ValueError("No data-collection available to save. Please run collect_data() first.")

            filename = f"{self.stock_ticker}_historical_data.csv"
            self.data.to_csv(filename)
            logging.info(f"Data for {self.stock_ticker} saved successfully to {filename}.")
            print(f"Data saved successfully to {filename}.")
        except Exception as e:
            logging.error(f"Error saving data-collection for {self.stock_ticker}: {e}")
            raise

def main():
    """
    Main entry point: prompts the user for a stock ticker and horizon,
    collects the data-collection, and saves it to a CSV file.
    """
    try:
        stock_ticker = input("Enter stock ticker: ").strip().upper()
        horizon_input = input("Enter horizon in years: ").strip()
        try:
            horizon = int(horizon_input)
        except ValueError:
            raise ValueError("Horizon must be an integer representing the number of years.")

        collector = StockDataCollector(stock_ticker, horizon)
        collector.collect_data()
        collector.save_data()
    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"Error in main: {e}")

if __name__ == "__main__":
    main()
