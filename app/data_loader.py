import pandas as pd
import numpy as np
import os

class DataLoader:
    def __init__(self, file_path: str = "data/Data.xlsx"):
        self.file_path = file_path
        self.fund_info = {}
        self.returns_df = None
        self.factor_df = None
        self.load_data()

    def load_data(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Data file {self.file_path} not found.")

        excel_file = pd.ExcelFile(self.file_path)
        
        df_info = pd.read_excel(excel_file, sheet_name="Fund Info")
        df_info['dividend_yield'] = df_info['dividend_yield'].fillna(0.0)
        
        for _, row in df_info.iterrows():
            self.fund_info[row['ticker']] = {
                'fund_name': row['fund_name'],
                'dividend_yield': float(row['dividend_yield'])
            }

        df_returns = pd.read_excel(excel_file, sheet_name="Fund Returns")
        self.returns_df = df_returns.pivot(index='date', columns='ticker', values='total_return').sort_index()

        df_factor = pd.read_excel(excel_file, sheet_name="Factor Returns")
        self.factor_df = df_factor.pivot(index='date', columns='index_ticker', values='total_return').sort_index()

    def get_fund_info(self, tickers: list[str]) -> dict:
        """Returns metadata for the requested tickers."""
        return {t: self.fund_info.get(t, {}) for t in tickers}

    def get_returns_data(self, tickers: list[str]) -> pd.DataFrame:
        """Returns a DataFrame of historical returns filtered by the requested tickers."""
        return self.returns_df[tickers].dropna()

    def get_factor_data(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Returns factor returns matching the exact dates provided."""
        return self.factor_df.reindex(dates).dropna()

data_store = DataLoader()
