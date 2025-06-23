import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalysis:
    """Handles PCA, clustering, and cointegration analysis."""
    
    def __init__(self):
        self.scaler = StandardScaler()
    
    def perform_pca(self, returns: pd.DataFrame, n_components: int) -> dict:
        """
        Perform Principal Component Analysis on stock returns.
        
        Args:
            returns: DataFrame of stock returns
            n_components: Number of principal components to compute
            
        Returns:
            Dictionary containing PCA results
        """
        # Standardize the data
        returns_scaled = self.scaler.fit_transform(returns.fillna(0))
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(returns_scaled)
        
        # Create results dictionary
        results = {
            'pca_model': pca,
            'components': components,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(pca.explained_variance_ratio_),
            'feature_names': returns.columns.tolist(),
            'loadings': pca.components_.T,  # Transpose to get features x components
            'scaled_returns': returns_scaled
        }
        
        return results
    
    def perform_clustering(self, returns: pd.DataFrame, n_clusters: int) -> dict:
        """
        Perform K-means clustering on stock returns.
        
        Args:
            returns: DataFrame of stock returns
            n_clusters: Number of clusters
            
        Returns:
            Dictionary containing clustering results
        """
        # Standardize the data
        returns_scaled = self.scaler.fit_transform(returns.fillna(0).T)  # Transpose to cluster stocks
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(returns_scaled)
        
        # Calculate cluster centers and distances
        cluster_centers = kmeans.cluster_centers_
        distances = kmeans.transform(returns_scaled)
        
        results = {
            'kmeans_model': kmeans,
            'cluster_labels': cluster_labels,
            'cluster_centers': cluster_centers,
            'distances_to_centers': distances,
            'inertia': kmeans.inertia_,
            'stock_names': returns.columns.tolist()
        }
        
        return results
    
    def calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix of stock returns."""
        correlation_matrix = returns.corr()
        return correlation_matrix
    
    def test_cointegration(self, price1: pd.Series, price2: pd.Series) -> dict:
        """
        Test for cointegration between two price series.
        
        Args:
            price1, price2: Price series for two stocks
            
        Returns:
            Dictionary with cointegration test results
        """
        try:
            # Align the series
            aligned_data = pd.concat([price1, price2], axis=1).dropna()
            if len(aligned_data) < 30:  # Need sufficient data
                return {'cointegrated': False, 'p_value': 1.0, 'error': 'Insufficient data'}
            
            series1 = aligned_data.iloc[:, 0]
            series2 = aligned_data.iloc[:, 1]
            
            # Perform Engle-Granger cointegration test
            score, p_value, crit_values = coint(series1, series2)
            
            # Additional check: test stationarity of residuals
            # Fit linear regression
            X = series2.values.reshape(-1, 1)
            y = series1.values
            
            # Add constant term
            X_with_const = np.column_stack([np.ones(len(X)), X])
            
            model = OLS(y, X_with_const).fit()
            residuals = model.resid
            
            # Test residuals for stationarity
            adf_stat, adf_p_value, _, _, _, _ = adfuller(residuals, maxlag=1)
            
            # Calculate hedge ratio (beta)
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            
            return {
                'cointegrated': p_value < 0.05,
                'p_value': p_value,
                'coint_score': score,
                'critical_values': crit_values,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'adf_statistic': adf_stat,
                'adf_p_value': adf_p_value,
                'residuals': residuals,
                'r_squared': model.rsquared
            }
            
        except Exception as e:
            return {'cointegrated': False, 'p_value': 1.0, 'error': str(e)}
    
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, lookback_window: int) -> list:
        """
        Find cointegrated pairs from a universe of stocks.
        
        Args:
            price_data: DataFrame of stock prices
            lookback_window: Number of days to use for cointegration testing
            
        Returns:
            List of dictionaries containing pair information
        """
        stocks = price_data.columns.tolist()
        cointegrated_pairs = []
        
        # Use recent data for cointegration testing
        recent_data = price_data.tail(lookback_window)
        
        # Test all pairs
        total_pairs = len(stocks) * (len(stocks) - 1) // 2
        pair_count = 0
        
        for i in range(len(stocks)):
            for j in range(i + 1, len(stocks)):
                stock1, stock2 = stocks[i], stocks[j]
                
                # Get price series
                price1 = recent_data[stock1].dropna()
                price2 = recent_data[stock2].dropna()
                
                if len(price1) < 30 or len(price2) < 30:
                    continue
                
                # Test cointegration
                coint_result = self.test_cointegration(price1, price2)
                
                if coint_result.get('cointegrated', False):
                    pair_info = {
                        'stock1': stock1,
                        'stock2': stock2,
                        'p_value': coint_result['p_value'],
                        'coint_score': coint_result['coint_score'],
                        'hedge_ratio': coint_result['hedge_ratio'],
                        'intercept': coint_result['intercept'],
                        'r_squared': coint_result['r_squared'],
                        'adf_p_value': coint_result['adf_p_value']
                    }
                    cointegrated_pairs.append(pair_info)
                
                pair_count += 1
        
        # Sort by p-value (most significant first)
        cointegrated_pairs.sort(key=lambda x: x['p_value'])
        
        return cointegrated_pairs
    
    def calculate_zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling z-score of spread."""
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion for a spread."""
        try:
            # Fit AR(1) model: spread_t = a + b * spread_{t-1} + error
            spread_lag = spread.shift(1).dropna()
            spread_current = spread[1:]
            
            # Align the series
            aligned_data = pd.concat([spread_current, spread_lag], axis=1).dropna()
            if len(aligned_data) < 10:
                return np.nan
            
            y = aligned_data.iloc[:, 0].values
            x = aligned_data.iloc[:, 1].values
            
            # Add constant
            X = np.column_stack([np.ones(len(x)), x])
            
            # Fit regression
            model = OLS(y, X).fit()
            beta = model.params[1]
            
            # Calculate half-life
            if beta >= 1 or beta <= 0:
                return np.nan
            
            half_life = -np.log(2) / np.log(beta)
            return half_life
            
        except Exception:
            return np.nan
