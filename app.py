import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from statistical_analysis import StatisticalAnalysis
from trading_strategies import TradingStrategies
from visualization import Visualization

# Page configuration
st.set_page_config(
    page_title="Statistical Arbitrage Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Title and description
st.title("📊 Statistical Arbitrage with PCA and Clustering")
st.markdown("""
This application uses Principal Component Analysis (PCA) and clustering techniques to identify 
cointegrated stock pairs and test long-short trading strategies on S&P 500 stocks.
""")

# Sidebar for parameters
st.sidebar.header("Configuration")

# Data parameters
st.sidebar.subheader("Data Settings")
num_stocks = st.sidebar.slider("Number of S&P 500 stocks to analyze", 50, 500, 100, 25)
period = st.sidebar.selectbox("Time period", ["1y", "2y", "3y", "5y"], index=1)

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
n_components = st.sidebar.slider("Number of PCA components", 2, 20, 5)
n_clusters = st.sidebar.slider("Number of clusters for K-means", 2, 15, 5)
lookback_window = st.sidebar.slider("Cointegration lookback window (days)", 30, 252, 60)

# Strategy parameters
st.sidebar.subheader("Trading Strategy")
entry_threshold = st.sidebar.slider("Entry Z-score threshold", 1.0, 3.0, 2.0, 0.1)
exit_threshold = st.sidebar.slider("Exit Z-score threshold", 0.1, 1.0, 0.5, 0.1)
stop_loss = st.sidebar.slider("Stop loss Z-score", 3.0, 5.0, 4.0, 0.1)

# Initialize components
@st.cache_resource
def initialize_components():
    data_loader = DataLoader()
    stat_analysis = StatisticalAnalysis()
    trading_strategies = TradingStrategies()
    visualization = Visualization()
    return data_loader, stat_analysis, trading_strategies, visualization

data_loader, stat_analysis, trading_strategies, visualization = initialize_components()

# Main application
def main():
    # Step 1: Load Data
    st.header("1. Data Loading")
    
    if st.button("Load S&P 500 Stock Data", type="primary"):
        with st.spinner(f"Loading data for {num_stocks} stocks..."):
            try:
                stock_data, stock_symbols = data_loader.load_sp500_data(num_stocks, period)
                st.session_state.stock_data = stock_data
                st.session_state.stock_symbols = stock_symbols
                st.session_state.data_loaded = True
                st.success(f"Successfully loaded data for {len(stock_symbols)} stocks")
                
                # Display basic statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Stocks Loaded", len(stock_symbols))
                with col2:
                    st.metric("Date Range", f"{stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
                with col3:
                    st.metric("Trading Days", len(stock_data))
                with col4:
                    st.metric("Missing Data %", f"{(stock_data.isnull().sum().sum() / (len(stock_data) * len(stock_symbols)) * 100):.2f}%")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                return
    
    if not st.session_state.data_loaded:
        st.info("Please load the stock data to continue with the analysis.")
        return
    
    # Step 2: Statistical Analysis
    st.header("2. Statistical Analysis")
    
    if st.button("Perform PCA and Clustering Analysis", type="primary"):
        with st.spinner("Performing statistical analysis..."):
            try:
                # Calculate returns
                returns = st.session_state.stock_data.pct_change().dropna()
                
                # PCA Analysis
                pca_results = stat_analysis.perform_pca(returns, n_components)
                st.session_state.pca_results = pca_results
                
                # Clustering Analysis
                cluster_results = stat_analysis.perform_clustering(returns, n_clusters)
                st.session_state.cluster_results = cluster_results
                
                # Correlation Analysis
                correlation_matrix = stat_analysis.calculate_correlation_matrix(returns)
                st.session_state.correlation_matrix = correlation_matrix
                
                # Find cointegrated pairs
                cointegrated_pairs = stat_analysis.find_cointegrated_pairs(
                    st.session_state.stock_data, 
                    lookback_window
                )
                st.session_state.cointegrated_pairs = cointegrated_pairs
                st.session_state.analysis_complete = True
                
                st.success("Statistical analysis completed successfully!")
                
            except Exception as e:
                st.error(f"Error in statistical analysis: {str(e)}")
                return
    
    if not st.session_state.analysis_complete:
        st.info("Please perform the statistical analysis to continue.")
        return
    
    # Step 3: Display Results
    st.header("3. Analysis Results")
    
    # PCA Results
    st.subheader("Principal Component Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Explained variance plot
        fig = visualization.plot_pca_explained_variance(st.session_state.pca_results)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # PCA components heatmap
        fig = visualization.plot_pca_components_heatmap(
            st.session_state.pca_results, 
            st.session_state.stock_symbols
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Clustering Results
    st.subheader("Clustering Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster visualization in PCA space
        fig = visualization.plot_clusters_pca_space(
            st.session_state.pca_results,
            st.session_state.cluster_results,
            st.session_state.stock_symbols
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cluster composition
        cluster_composition = visualization.get_cluster_composition(
            st.session_state.cluster_results,
            st.session_state.stock_symbols
        )
        st.dataframe(cluster_composition, use_container_width=True)
    
    # Correlation Matrix
    st.subheader("Correlation Matrix")
    fig = visualization.plot_correlation_heatmap(st.session_state.correlation_matrix)
    st.plotly_chart(fig, use_container_width=True)
    
    # Cointegrated Pairs
    st.subheader("Cointegrated Pairs")
    if len(st.session_state.cointegrated_pairs) > 0:
        pairs_df = pd.DataFrame(st.session_state.cointegrated_pairs)
        pairs_df = pairs_df.sort_values('p_value').head(20)  # Show top 20 pairs
        st.dataframe(pairs_df, use_container_width=True)
        
        # Step 4: Trading Strategy
        st.header("4. Trading Strategy Backtesting")
        
        # Select pair for strategy testing
        selected_pair = st.selectbox(
            "Select a cointegrated pair for strategy testing:",
            options=range(len(pairs_df)),
            format_func=lambda x: f"{pairs_df.iloc[x]['stock1']} - {pairs_df.iloc[x]['stock2']} (p-value: {pairs_df.iloc[x]['p_value']:.4f})"
        )
        
        if st.button("Backtest Trading Strategy", type="primary"):
            with st.spinner("Backtesting trading strategy..."):
                try:
                    pair_info = pairs_df.iloc[selected_pair]
                    stock1, stock2 = pair_info['stock1'], pair_info['stock2']
                    
                    # Get price data for the pair
                    price1 = st.session_state.stock_data[stock1].dropna()
                    price2 = st.session_state.stock_data[stock2].dropna()
                    
                    # Align the data
                    common_dates = price1.index.intersection(price2.index)
                    price1 = price1[common_dates]
                    price2 = price2[common_dates]
                    
                    # Run backtest
                    backtest_results = trading_strategies.backtest_pairs_strategy(
                        price1, price2, stock1, stock2,
                        entry_threshold, exit_threshold, stop_loss
                    )
                    
                    st.session_state.backtest_results = backtest_results
                    
                    # Display strategy performance
                    st.subheader("Strategy Performance")
                    
                    # Performance metrics
                    metrics = backtest_results['metrics']
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{metrics['total_return']:.2%}")
                    with col2:
                        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
                    with col3:
                        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
                    with col4:
                        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
                    
                    # Strategy visualization
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Cumulative returns
                        fig = visualization.plot_strategy_performance(backtest_results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Spread and Z-score
                        fig = visualization.plot_spread_and_zscore(backtest_results)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Residuals analysis
                    st.subheader("Mean Reversion Analysis")
                    fig = visualization.plot_residuals_analysis(backtest_results)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Trade details
                    st.subheader("Trade History")
                    if len(backtest_results['trades']) > 0:
                        trades_df = pd.DataFrame(backtest_results['trades'])
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.info("No trades were executed with the current parameters.")
                        
                except Exception as e:
                    st.error(f"Error in backtesting: {str(e)}")
    else:
        st.warning("No cointegrated pairs found with the current parameters. Try adjusting the lookback window or significance level.")

# Run the main application
if __name__ == "__main__":
    main()
