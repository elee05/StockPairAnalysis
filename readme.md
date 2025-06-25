# Statistical Arbitrage Application

## Overview

This is a fully functional Streamlit-based statistical arbitrage application that uses Principal Component Analysis (PCA) and clustering techniques to identify cointegrated stock pairs and test long-short trading strategies on S&P 500 stocks. The application provides an interactive web interface for financial analysis and backtesting of statistical arbitrage strategies.

## Running
- streamlit python app.py
- open https://localhost5000

## Data Flow

1. **Data Ingestion**: Application fetches S&P 500 symbols from Wikipedia and historical price data from Yahoo Finance
2. **Data Preprocessing**: Returns calculation, standardization, and missing data handling
3. **Statistical Analysis**: PCA decomposition and K-means clustering on processed returns
4. **Pair Identification**: Cointegration testing within clusters to identify tradeable pairs
5. **Strategy Implementation**: Signal generation and backtesting on identified pairs
6. **Visualization**: Interactive charts displaying analysis results and strategy performance


## Key Components
Fetches S&P 500 stock symbols and historical price data from Wikipedia for S&P 500 list, Yahoo Finance for price data however, predefined list of major stocks if web scraping fails

- **PCA Implementation**: Dimensionality reduction on stock returns
- **Clustering**: K-means clustering for stock grouping
- **Cointegration Testing**: Statistical tests for pair relationships

### 3. Trading Strategies
- **Spread Calculation**: Computes spreads between cointegrated pairs
- **Signal Generation**: Z-score based entry/exit signals
- **Risk Management**: Stop-loss and position sizing logic
- **Backtesting Framework**: Historical strategy performance evaluation

### 4. Visualization 
- **PCA Visualization**: Explained variance and component analysis plots
- **Strategy Performance**: Equity curves and drawdown analysis
- **Pair Analysis**: Spread and correlation visualizations



## External Dependencies

### Data Sources
- **Wikipedia**: S&P 500 company list scraping
- **Yahoo Finance**: Historical stock price data via yfinance library

### Python Libraries
- **Core**: pandas, numpy for data manipulation
- **ML/Stats**: scikit-learn, statsmodels for statistical analysis
- **Visualization**: plotly, seaborn for interactive charts
- **Web Framework**: streamlit for UI
- **Data Fetching**: yfinance, requests for external data


