# Statistical Arbitrage Application

## Overview

This is a fully functional Streamlit-based statistical arbitrage application that uses Principal Component Analysis (PCA) and clustering techniques to identify cointegrated stock pairs and test long-short trading strategies on S&P 500 stocks. The application provides an interactive web interface for financial analysis and backtesting of statistical arbitrage strategies.

## Running
- streamlit python app.py
- open https://localhost5000


## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web UI and interactive components
- **Visualization**: Plotly for interactive charts and graphs
- **Layout**: Wide layout with expandable sidebar for parameter configuration
- **State Management**: Streamlit session state for maintaining analysis results across interactions

### Backend Architecture
- **Modular Design**: Separate classes for different functionalities (DataLoader, StatisticalAnalysis, TradingStrategies, Visualization)
- **Data Processing**: Pandas and NumPy for data manipulation and numerical computations
- **Statistical Analysis**: Scikit-learn for PCA and clustering, Statsmodels for cointegration testing
- **Financial Data**: yfinance for real-time stock data retrieval

## Key Components

### 1. Data Loading (`DataLoader`)
- **Purpose**: Fetches S&P 500 stock symbols and historical price data
- **Data Source**: Wikipedia for S&P 500 list, Yahoo Finance for price data
- **Fallback Strategy**: Predefined list of major stocks if web scraping fails
- **Data Cleaning**: Handles symbol formatting and missing data

### 2. Statistical Analysis (`StatisticalAnalysis`)
- **PCA Implementation**: Dimensionality reduction on stock returns
- **Clustering**: K-means clustering for stock grouping
- **Cointegration Testing**: Statistical tests for pair relationships
- **Data Preprocessing**: Standardization and normalization

### 3. Trading Strategies (`TradingStrategies`)
- **Spread Calculation**: Computes spreads between cointegrated pairs
- **Signal Generation**: Z-score based entry/exit signals
- **Risk Management**: Stop-loss and position sizing logic
- **Backtesting Framework**: Historical strategy performance evaluation

### 4. Visualization (`Visualization`)
- **Interactive Charts**: Plotly-based visualizations for analysis results
- **PCA Visualization**: Explained variance and component analysis plots
- **Strategy Performance**: Equity curves and drawdown analysis
- **Pair Analysis**: Spread and correlation visualizations

## Data Flow

1. **Data Ingestion**: Application fetches S&P 500 symbols from Wikipedia and historical price data from Yahoo Finance
2. **Data Preprocessing**: Returns calculation, standardization, and missing data handling
3. **Statistical Analysis**: PCA decomposition and K-means clustering on processed returns
4. **Pair Identification**: Cointegration testing within clusters to identify tradeable pairs
5. **Strategy Implementation**: Signal generation and backtesting on identified pairs
6. **Visualization**: Interactive charts displaying analysis results and strategy performance

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

### Configuration Management
- **Package Management**: uv with pyproject.toml for dependency resolution
- **Streamlit Config**: Custom server configuration in .streamlit/config.toml

## Deployment Strategy

### Platform
- **Runtime**: Python 3.11 with Nix package management
- **Port Configuration**: Streamlit server on port 5000


### Workflow Setup
- **Run Button**: Configured for easy one-click deployment
- **Process Monitoring**: Port waiting and parallel task execution
- **Development**: Hot reload enabled for iterative development

