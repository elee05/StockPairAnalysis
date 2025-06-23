import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, List
import requests
from io import StringIO
import time

class DataLoader:
    """Handles data loading and preprocessing for statistical arbitrage analysis."""
    
    def __init__(self):
        self.sp500_symbols = None
    
    def get_sp500_symbols(self) -> List[str]:
        """Fetch S&P 500 symbols from Wikipedia."""
        try:
            # Try to get S&P 500 list from Wikipedia
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            symbols = sp500_table['Symbol'].tolist()
            
            # Clean symbols (remove dots and other special characters that might cause issues)
            cleaned_symbols = []
            for symbol in symbols:
                # Replace dots with dashes (common yfinance convention)
                cleaned_symbol = symbol.replace('.', '-')
                cleaned_symbols.append(cleaned_symbol)
            
            return cleaned_symbols[:500]  # Ensure we don't exceed 500
            
        except Exception as e:
            st.warning(f"Could not fetch S&P 500 list from Wikipedia: {str(e)}")
            # Fallback to a predefined list of major S&P 500 stocks
            return self._get_fallback_symbols()
    
    def _get_fallback_symbols(self) -> List[str]:
        """Fallback list of major S&P 500 stocks."""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
            'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'PFE',
            'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
            'CRM', 'ACN', 'VZ', 'ADBE', 'NFLX', 'CMCSA', 'NKE', 'DHR', 'TXN',
            'NEE', 'RTX', 'QCOM', 'PM', 'UNP', 'T', 'LIN', 'SPGI', 'HON',
            'COP', 'LOW', 'INTU', 'IBM', 'GS', 'AMGN', 'CAT', 'AMD', 'BKNG',
            'BLK', 'DE', 'AXP', 'GILD', 'MDT', 'TGT', 'LRCX', 'SYK', 'MU',
            'CVS', 'TMUS', 'CI', 'REGN', 'PYPL', 'MDLZ', 'SO', 'PLD', 'ZTS',
            'ISRG', 'CB', 'DUK', 'C', 'MMM', 'SCHW', 'ITW', 'TJX', 'BSX',
            'MO', 'HUM', 'SLB', 'EOG', 'BDX', 'AON', 'USB', 'ICE', 'EQIX',
            'WM', 'GE', 'APD', 'CL', 'NSC', 'EMR', 'FCX', 'PNC', 'MSCI',
            'MCD', 'CSX', 'FIS', 'MMC', 'TFC', 'GM', 'F', 'ADSK', 'ECL'
        ]
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_sp500_data(_self, num_stocks: int, period: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load S&P 500 stock data using yfinance.
        
        Args:
            num_stocks: Number of stocks to load
            period: Time period ('1y', '2y', '3y', '5y')
            
        Returns:
            Tuple of (price_data, stock_symbols)
        """
        # Get S&P 500 symbols
        if _self.sp500_symbols is None:
            _self.sp500_symbols = _self.get_sp500_symbols()
        
        # Select subset of symbols
        selected_symbols = _self.sp500_symbols[:num_stocks]
        
        # Filter out delisted symbols (like ATVI)
        excluded_symbols = ['ATVI']  # Add known delisted symbols
        selected_symbols = [s for s in selected_symbols if s not in excluded_symbols]
        
        # Download data
        successful_symbols = []
        price_data_list = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, symbol in enumerate(selected_symbols):
            try:
                status_text.text(f"Loading {symbol}... ({i+1}/{len(selected_symbols)})")
                
                # Download data for this symbol with timeout and retry
                ticker = yf.Ticker(symbol)
                hist = None
                max_retries = 2
                
                for retry in range(max_retries):
                    try:
                        hist = ticker.history(period=period, timeout=15)
                        if len(hist) > 0:
                            break
                        else:
                            time.sleep(1)  # Brief pause before retry
                    except Exception as retry_error:
                        if retry == max_retries - 1:
                            raise retry_error
                        time.sleep(2)  # Longer pause before retry
                
                if len(hist) > 0 and not hist['Close'].isnull().all():
                    # Use adjusted close price
                    price_series = hist['Close'].rename(symbol)
                    price_data_list.append(price_series)
                    successful_symbols.append(symbol)
                    if len(successful_symbols) <= 5:  # Show first few successes
                        st.info(f"✓ Successfully loaded {symbol}")
                else:
                    st.warning(f"No data available for {symbol}")
                
                progress_bar.progress((i + 1) / len(selected_symbols))
                
            except Exception as e:
                error_details = str(e)
                if "connection" in error_details.lower():
                    st.error(f"Network error for {symbol}: {error_details}")
                elif "timeout" in error_details.lower():
                    st.warning(f"Timeout loading {symbol} - trying next stock")
                else:
                    st.warning(f"Failed to load data for {symbol}: {error_details}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        
        st.info(f"Successfully loaded {len(successful_symbols)} out of {len(selected_symbols)} stocks")
        
        if len(successful_symbols) == 0:
            error_msg = f"No stock data could be loaded successfully. Attempted to load {len(selected_symbols)} symbols. "
            error_msg += "This could be due to: 1) Network connectivity issues, 2) Yahoo Finance API blocking requests, "
            error_msg += "3) All symbols being invalid/delisted, or 4) yfinance library issues. "
            error_msg += "Try: reducing the number of stocks, checking internet connection, or running with a VPN."
            raise ValueError(error_msg)
        
        # Combine all price data
        price_data = pd.concat(price_data_list, axis=1)
        
        # Forward fill and backward fill missing values
        price_data = price_data.ffill().bfill()
        
        # Remove stocks with too much missing data
        missing_threshold = 0.1  # Allow up to 10% missing data
        valid_stocks = []
        for symbol in successful_symbols:
            missing_pct = price_data[symbol].isnull().sum() / len(price_data)
            if missing_pct <= missing_threshold:
                valid_stocks.append(symbol)
        
        price_data = price_data[valid_stocks]
        
        return price_data, valid_stocks
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the loaded data for quality issues."""
        if data.empty:
            return False
        
        # Check for excessive missing values
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.2:  # More than 20% missing
            return False
        
        # Check for sufficient history
        if len(data) < 50:  # Less than 50 trading days
            return False
        
        # Check for price validity (no negative prices, reasonable price ranges)
        if (data <= 0).any().any():
            return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data for analysis."""
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Ensure all prices are positive
        data = data[data > 0]
        
        # Sort by date
        data = data.sort_index()
        
        return data


# import yfinance as yf
# import pandas as pd
# import numpy as np
# import streamlit as st
# from typing import Tuple, List
# import requests
# from io import StringIO

# class DataLoader:
#     """Handles data loading and preprocessing for statistical arbitrage analysis."""
    
#     def __init__(self):
#         self.sp500_symbols = None
    
#     def get_sp500_symbols(self) -> List[str]:
#         """Fetch S&P 500 symbols from Wikipedia."""
#         try:
#             # Try to get S&P 500 list from Wikipedia
#             url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
#             tables = pd.read_html(url)
#             sp500_table = tables[0]
#             symbols = sp500_table['Symbol'].tolist()
            
#             # Clean symbols (remove dots and other special characters that might cause issues)
#             cleaned_symbols = []
#             for symbol in symbols:
#                 # Replace dots with dashes (common yfinance convention)
#                 cleaned_symbol = symbol.replace('.', '-')
#                 cleaned_symbols.append(cleaned_symbol)
            
#             return cleaned_symbols[:500]  # Ensure we don't exceed 500
            
#         except Exception as e:
#             st.warning(f"Could not fetch S&P 500 list from Wikipedia: {str(e)}")
#             # Fallback to a predefined list of major S&P 500 stocks
#             return self._get_fallback_symbols()
    
#     def _get_fallback_symbols(self) -> List[str]:
#         """Fallback list of major S&P 500 stocks."""
#         return [
#             'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK-B',
#             'UNH', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'PFE',
#             'BAC', 'KO', 'AVGO', 'PEP', 'TMO', 'COST', 'WMT', 'DIS', 'ABT',
#             'CRM', 'ACN', 'VZ', 'ADBE', 'NFLX', 'CMCSA', 'NKE', 'DHR', 'TXN',
#             'NEE', 'RTX', 'QCOM', 'PM', 'UNP', 'T', 'LIN', 'SPGI', 'HON',
#             'COP', 'LOW', 'INTU', 'IBM', 'GS', 'AMGN', 'CAT', 'AMD', 'BKNG',
#             'BLK', 'DE', 'AXP', 'GILD', 'MDT', 'TGT', 'LRCX', 'SYK', 'MU',
#             'CVS', 'TMUS', 'CI', 'REGN', 'PYPL', 'MDLZ', 'SO', 'PLD', 'ZTS',
#             'ISRG', 'CB', 'DUK', 'C', 'MMM', 'SCHW', 'ITW', 'TJX', 'BSX',
#             'MO', 'HUM', 'SLB', 'EOG', 'BDX', 'AON', 'USB', 'ICE', 'EQIX',
#             'WM', 'GE', 'APD', 'CL', 'NSC', 'EMR', 'FCX', 'PNC', 'MSCI',
#             'MCD', 'CSX', 'FIS', 'MMC', 'TFC', 'GM', 'F', 'ADSK', 'ECL'
#         ]
    
#     @st.cache_data(ttl=3600)  # Cache for 1 hour
#     def load_sp500_data(_self, num_stocks: int, period: str) -> Tuple[pd.DataFrame, List[str]]:
#         """
#         Load S&P 500 stock data using yfinance.
        
#         Args:
#             num_stocks: Number of stocks to load
#             period: Time period ('1y', '2y', '3y', '5y')
            
#         Returns:
#             Tuple of (price_data, stock_symbols)
#         """
#         # Get S&P 500 symbols
#         if _self.sp500_symbols is None:
#             _self.sp500_symbols = _self.get_sp500_symbols()
        
#         # Select subset of symbols
#         selected_symbols = _self.sp500_symbols[:num_stocks]
        
#         # Filter out delisted symbols (like ATVI)
#         excluded_symbols = ['ATVI']  # Add known delisted symbols
#         selected_symbols = [s for s in selected_symbols if s not in excluded_symbols]
        
#         # Download data
#         successful_symbols = []
#         price_data_list = []
        
#         progress_bar = st.progress(0)
#         status_text = st.empty()
        
#         for i, symbol in enumerate(selected_symbols):
#             try:
#                 status_text.text(f"Loading {symbol}... ({i+1}/{len(selected_symbols)})")
                
#                 # Download data for this symbol
#                 ticker = yf.Ticker(symbol)
#                 hist = ticker.history(period=period)
                
#                 if len(hist) > 0 and not hist['Close'].isnull().all():
#                     # Use adjusted close price
#                     price_series = hist['Close'].rename(symbol)
#                     price_data_list.append(price_series)
#                     successful_symbols.append(symbol)
                
#                 progress_bar.progress((i + 1) / len(selected_symbols))
                
#             except Exception as e:
#                 st.warning(f"Failed to load data for {symbol}: {str(e)}")
#                 continue
        
#         progress_bar.empty()
#         status_text.empty()
        
#         if len(successful_symbols) == 0:
#             raise ValueError("No stock data could be loaded successfully")
        
#         # Combine all price data
#         price_data = pd.concat(price_data_list, axis=1)
        
#         # Forward fill and backward fill missing values
#         price_data = price_data.ffill().bfill()
        
#         # Remove stocks with too much missing data
#         missing_threshold = 0.1  # Allow up to 10% missing data
#         valid_stocks = []
#         for symbol in successful_symbols:
#             missing_pct = price_data[symbol].isnull().sum() / len(price_data)
#             if missing_pct <= missing_threshold:
#                 valid_stocks.append(symbol)
        
#         price_data = price_data[valid_stocks]
        
#         return price_data, valid_stocks
    
#     def validate_data(self, data: pd.DataFrame) -> bool:
#         """Validate the loaded data for quality issues."""
#         if data.empty:
#             return False
        
#         # Check for excessive missing values
#         missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
#         if missing_pct > 0.2:  # More than 20% missing
#             return False
        
#         # Check for sufficient history
#         if len(data) < 50:  # Less than 50 trading days
#             return False
        
#         # Check for price validity (no negative prices, reasonable price ranges)
#         if (data <= 0).any().any():
#             return False
        
#         return True
    
#     def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Preprocess the data for analysis."""
#         # Remove any remaining NaN values
#         data = data.dropna()
        
#         # Ensure all prices are positive
#         data = data[data > 0]
        
#         # Sort by date
#         data = data.sort_index()
        
#         return data
