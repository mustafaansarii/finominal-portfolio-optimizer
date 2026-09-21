import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path: str = "data/Data.xlsx"):
        self.file_path = file_path
        
        # In-memory storage for our data
        self.fund_info = {}       # Dict: ticker -> { 'fund_name': str, 'dividend_yield': float }
        self.returns_df = None    # DataFrame: Date index, Ticker columns
        self.factor_df = None     # DataFrame: Date index, Factor columns
        
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        excel_file = pd.ExcelFile(self.file_path)
        
        # 1. Load Fund Info
        df_info = pd.read_excel(excel_file, sheet_name="Fund Info")
        # GLD might have NaN for dividend_yield, we'll fill with 0
        df_info['dividend_yield'] = df_info['dividend_yield'].fillna(0.0)
        
        for _, row in df_info.iterrows():
            self.fund_info[row['ticker']] = {
                'fund_name': row['fund_name'],
                'dividend_yield': float(row['dividend_yield'])
            }

        # 2. Load Fund Returns
        df_returns = pd.read_excel(excel_file, sheet_name="Fund Returns")
        # Pivot the dataframe so each ticker is a column, and dates are the index
        self.returns_df = df_returns.pivot(index='date', columns='ticker', values='total_return')
        # Sort by date ascending to ensure chronological order
        self.returns_df = self.returns_df.sort_index()

        # 3. Load Factor Returns
        df_factor = pd.read_excel(excel_file, sheet_name="Factor Returns")
        # Pivot similarly
        self.factor_df = df_factor.pivot(index='date', columns='index_ticker', values='total_return')
        self.factor_df = self.factor_df.sort_index()

    def get_fund_info(self, tickers: list[str]) -> dict:
        """Returns metadata for the requested tickers."""
        return {t: self.fund_info.get(t, {}) for t in tickers}

    def get_returns_data(self, tickers: list[str]) -> pd.DataFrame:
        """Returns a DataFrame of historical returns filtered by the requested tickers."""
        # Drop rows where any of the requested tickers have missing data
        # so we calculate covariance on a common timeframe.
        df_filtered = self.returns_df[tickers].dropna()
        return df_filtered

    def get_factor_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Returns factor returns matching the exact dates provided."""
        # We find the intersection of dates between the portfolio returns and factor returns
        return self.factor_df.reindex(dates).dropna()


# Create a global instance to be loaded once at startup
data_store = DataLoader()
