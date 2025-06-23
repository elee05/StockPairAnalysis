import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
from typing import Dict, List

class Visualization:
    """Handles all visualization tasks for the statistical arbitrage application."""
    
    def __init__(self):
        # Set default color palette
        self.colors = px.colors.qualitative.Set1
    
    def plot_pca_explained_variance(self, pca_results: Dict) -> go.Figure:
        """Plot PCA explained variance ratio."""
        explained_var = pca_results['explained_variance_ratio']
        cumulative_var = pca_results['cumulative_variance_ratio']
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Individual Explained Variance', 'Cumulative Explained Variance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Individual explained variance
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var,
                name='Individual',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Cumulative explained variance
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Principal Component", row=1, col=1)
        fig.update_xaxes(title_text="Principal Component", row=1, col=2)
        fig.update_yaxes(title_text="Explained Variance Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative Variance Ratio", row=1, col=2)
        
        fig.update_layout(
            title="PCA Explained Variance Analysis",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_pca_components_heatmap(self, pca_results: Dict, stock_symbols: List[str]) -> go.Figure:
        """Plot PCA components as a heatmap."""
        loadings = pca_results['loadings']
        n_components = min(5, loadings.shape[1])  # Show top 5 components
        
        # Select top stocks by absolute loading for visualization
        top_stocks_idx = np.argsort(np.abs(loadings).sum(axis=1))[-20:]  # Top 20 stocks
        
        loadings_subset = loadings[top_stocks_idx, :n_components]
        stocks_subset = [stock_symbols[i] for i in top_stocks_idx]
        
        fig = go.Figure(data=go.Heatmap(
            z=loadings_subset,
            x=[f'PC{i+1}' for i in range(n_components)],
            y=stocks_subset,
            colorscale='RdBu',
            zmid=0,
            text=np.round(loadings_subset, 3),
            texttemplate="%{text}",
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="PCA Component Loadings (Top 20 Stocks)",
            xaxis_title="Principal Components",
            yaxis_title="Stocks",
            height=600
        )
        
        return fig
    
    def plot_clusters_pca_space(self, pca_results: Dict, cluster_results: Dict, 
                               stock_symbols: List[str]) -> go.Figure:
        """Plot clusters in PCA space."""
        components = pca_results['components']
        cluster_labels = cluster_results['cluster_labels']
        
        # Use first two principal components
        pc1 = components[:, 0]
        pc2 = components[:, 1]
        
        # Ensure we have the correct number of stock symbols
        # In case there's a mismatch, use the minimum length
        min_length = min(len(cluster_labels), len(stock_symbols), len(pc1))
        cluster_labels = cluster_labels[:min_length]
        stock_symbols = stock_symbols[:min_length]
        pc1 = pc1[:min_length]
        pc2 = pc2[:min_length]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot each cluster with different colors
        unique_labels = np.unique(cluster_labels)
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            mask_indices = np.where(mask)[0]
            fig.add_trace(go.Scatter(
                x=pc1[mask],
                y=pc2[mask],
                mode='markers',
                name=f'Cluster {label}',
                text=[stock_symbols[j] for j in mask_indices],
                hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>',
                marker=dict(
                    size=8,
                    color=self.colors[i % len(self.colors)],
                    opacity=0.7
                )
            ))
        
        # Add cluster centers
        centers_pca = cluster_results['kmeans_model'].transform(pca_results['scaled_returns'].T)
        fig.add_trace(go.Scatter(
            x=centers_pca[:, 0],
            y=centers_pca[:, 1],
            mode='markers',
            name='Cluster Centers',
            marker=dict(
                size=15,
                color='black',
                symbol='x',
                line=dict(width=2)
            ),
            hovertemplate='Cluster Center<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
        
        explained_var = pca_results['explained_variance_ratio']
        fig.update_layout(
            title="Stock Clusters in PCA Space",
            xaxis_title=f"First Principal Component ({explained_var[0]:.1%} variance)",
            yaxis_title=f"Second Principal Component ({explained_var[1]:.1%} variance)",
            height=500
        )
        
        return fig
    
    def get_cluster_composition(self, cluster_results: Dict, stock_symbols: List[str]) -> pd.DataFrame:
        """Get cluster composition as a DataFrame."""
        cluster_labels = cluster_results['cluster_labels']
        
        # Create DataFrame
        df = pd.DataFrame({
            'Stock': stock_symbols,
            'Cluster': cluster_labels
        })
        
        # Add cluster statistics
        cluster_stats = []
        for cluster_id in np.unique(cluster_labels):
            cluster_stocks = df[df['Cluster'] == cluster_id]['Stock'].tolist()
            cluster_stats.append({
                'Cluster': cluster_id,
                'Size': len(cluster_stocks),
                'Stocks': ', '.join(cluster_stocks[:5]) + ('...' if len(cluster_stocks) > 5 else '')
            })
        
        return pd.DataFrame(cluster_stats)
    
    def plot_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Plot correlation matrix as heatmap."""
        # Limit to top correlated stocks for visibility
        n_stocks = min(30, len(correlation_matrix))
        corr_subset = correlation_matrix.iloc[:n_stocks, :n_stocks]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_subset.values,
            x=corr_subset.columns,
            y=corr_subset.index,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f"Stock Returns Correlation Matrix (Top {n_stocks} stocks)",
            height=600,
            xaxis={'tickangle': 45},
            yaxis={'tickangle': 0}
        )
        
        return fig
    
    def plot_strategy_performance(self, backtest_results: Dict) -> go.Figure:
        """Plot strategy performance over time."""
        cumulative_returns = backtest_results['cumulative_returns']
        
        fig = go.Figure()
        
        # Strategy performance
        fig.add_trace(go.Scatter(
            x=cumulative_returns.index,
            y=(cumulative_returns - 1) * 100,
            mode='lines',
            name='Strategy Returns',
            line=dict(color='blue', width=2),
            hovertemplate='Date: %{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title="Cumulative Strategy Returns",
            xaxis_title="Date",
            yaxis_title="Cumulative Return (%)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_spread_and_zscore(self, backtest_results: Dict) -> go.Figure:
        """Plot spread and z-score with trading signals."""
        spread = backtest_results['spread']
        zscore = backtest_results['zscore']
        signals = backtest_results['signals']
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Spread', 'Z-Score and Trading Signals'),
            vertical_spacing=0.1,
            shared_xaxes=True
        )
        
        # Plot spread
        fig.add_trace(
            go.Scatter(
                x=spread.index,
                y=spread,
                mode='lines',
                name='Spread',
                line=dict(color='blue'),
                hovertemplate='Date: %{x}<br>Spread: %{y:.3f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add spread mean line
        fig.add_hline(y=spread.mean(), line_dash="dash", line_color="gray", 
                     opacity=0.5, row=1, col=1)
        
        # Plot z-score
        fig.add_trace(
            go.Scatter(
                x=zscore.index,
                y=zscore,
                mode='lines',
                name='Z-Score',
                line=dict(color='red'),
                hovertemplate='Date: %{x}<br>Z-Score: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Add threshold lines
        fig.add_hline(y=2.0, line_dash="dot", line_color="green", opacity=0.7, row=2, col=1)
        fig.add_hline(y=-2.0, line_dash="dot", line_color="green", opacity=0.7, row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # Add trading signals as markers
        long_signals = signals[signals == 1]
        short_signals = signals[signals == -1]
        
        if len(long_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=long_signals.index,
                    y=zscore[long_signals.index],
                    mode='markers',
                    name='Long Entry',
                    marker=dict(color='green', size=8, symbol='triangle-up'),
                    hovertemplate='Long Entry<br>Date: %{x}<br>Z-Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        if len(short_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=short_signals.index,
                    y=zscore[short_signals.index],
                    mode='markers',
                    name='Short Entry',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                    hovertemplate='Short Entry<br>Date: %{x}<br>Z-Score: %{y:.3f}<extra></extra>'
                ),
                row=2, col=1
            )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Spread", row=1, col=1)
        fig.update_yaxes(title_text="Z-Score", row=2, col=1)
        
        fig.update_layout(
            title=f"Spread Analysis: {backtest_results['stock1_name']} - {backtest_results['stock2_name']}",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_residuals_analysis(self, backtest_results: Dict) -> go.Figure:
        """Plot residuals analysis for mean reversion behavior."""
        spread = backtest_results['spread']
        zscore = backtest_results['zscore']
        half_life = backtest_results['spread_stats']['half_life']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Spread Distribution',
                'Z-Score Distribution', 
                'Spread Autocorrelation',
                'Mean Reversion Pattern'
            ),
            specs=[[{}, {}], [{}, {}]]
        )
        
        # Spread histogram
        fig.add_trace(
            go.Histogram(
                x=spread.dropna(),
                nbinsx=50,
                name='Spread',
                marker_color='lightblue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Z-score histogram
        fig.add_trace(
            go.Histogram(
                x=zscore.dropna(),
                nbinsx=50,
                name='Z-Score',
                marker_color='lightcoral',
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Autocorrelation
        if len(spread.dropna()) > 20:
            lags = range(1, min(21, len(spread.dropna()) // 4))
            autocorr = [spread.autocorr(lag=lag) for lag in lags]
            
            fig.add_trace(
                go.Bar(
                    x=list(lags),
                    y=autocorr,
                    name='Autocorrelation',
                    marker_color='lightgreen'
                ),
                row=2, col=1
            )
        
        # Mean reversion pattern
        spread_changes = spread.diff()
        spread_levels = spread.shift(1)
        
        # Remove NaN values
        valid_idx = ~(spread_changes.isna() | spread_levels.isna())
        if valid_idx.sum() > 10:
            fig.add_trace(
                go.Scatter(
                    x=spread_levels[valid_idx],
                    y=spread_changes[valid_idx],
                    mode='markers',
                    name='Mean Reversion',
                    marker=dict(color='purple', size=3, opacity=0.6),
                    hovertemplate='Spread Level: %{x:.3f}<br>Change: %{y:.3f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            # Add trend line
            z = np.polyfit(spread_levels[valid_idx], spread_changes[valid_idx], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(spread_levels[valid_idx].min(), spread_levels[valid_idx].max(), 100)
            
            fig.add_trace(
                go.Scatter(
                    x=x_trend,
                    y=p(x_trend),
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                ),
                row=2, col=2
            )
        
        fig.update_xaxes(title_text="Spread Value", row=1, col=1)
        fig.update_xaxes(title_text="Z-Score", row=1, col=2)
        fig.update_xaxes(title_text="Lag", row=2, col=1)
        fig.update_xaxes(title_text="Spread Level (t-1)", row=2, col=2)
        
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Correlation", row=2, col=1)
        fig.update_yaxes(title_text="Spread Change (t)", row=2, col=2)
        
        title = f"Mean Reversion Analysis"
        if not np.isnan(half_life):
            title += f" (Half-life: {half_life:.1f} days)"
        
        fig.update_layout(
            title=title,
            height=700,
            showlegend=False
        )
        
        return fig
