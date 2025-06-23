import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class TradingStrategies:
    """Implements statistical arbitrage trading strategies."""
    
    def __init__(self):
        pass
    
    def calculate_spread(self, price1: pd.Series, price2: pd.Series, hedge_ratio: float, intercept: float = 0) -> pd.Series:
        """Calculate spread between two price series."""
        # Align the series
        aligned_data = pd.concat([price1, price2], axis=1).dropna()
        p1, p2 = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
        
        # Calculate spread: price1 - hedge_ratio * price2 - intercept
        spread = p1 - hedge_ratio * p2 - intercept
        return spread
    
    def calculate_zscore(self, spread: pd.Series, window: int = 60) -> pd.Series:
        """Calculate rolling z-score of spread."""
        rolling_mean = spread.rolling(window=window, min_periods=20).mean()
        rolling_std = spread.rolling(window=window, min_periods=20).std()
        
        # Avoid division by zero
        rolling_std = rolling_std.replace(0, np.nan)
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def generate_signals(self, zscore: pd.Series, entry_threshold: float, 
                        exit_threshold: float, stop_loss: float) -> pd.Series:
        """
        Generate trading signals based on z-score thresholds.
        
        Returns:
            Series with values: 1 (long spread), -1 (short spread), 0 (no position)
        """
        signals = pd.Series(0, index=zscore.index)
        position = 0
        
        for i, z in enumerate(zscore):
            if pd.isna(z):
                signals.iloc[i] = position
                continue
            
            if position == 0:  # No current position
                if z > entry_threshold:
                    position = -1  # Short spread (sell stock1, buy stock2)
                elif z < -entry_threshold:
                    position = 1   # Long spread (buy stock1, sell stock2)
            
            elif position == 1:  # Long spread position
                if z > -exit_threshold or z < -stop_loss:
                    position = 0  # Exit position
            
            elif position == -1:  # Short spread position
                if z < exit_threshold or z > stop_loss:
                    position = 0  # Exit position
            
            signals.iloc[i] = position
        
        return signals
    
    def calculate_position_returns(self, price1: pd.Series, price2: pd.Series, 
                                 signals: pd.Series, hedge_ratio: float) -> pd.Series:
        """Calculate returns from the pairs trading strategy."""
        # Calculate returns for each stock
        returns1 = price1.pct_change()
        returns2 = price2.pct_change()
        
        # Calculate strategy returns
        # When signal = 1 (long spread): long stock1, short stock2
        # When signal = -1 (short spread): short stock1, long stock2
        strategy_returns = signals.shift(1) * (returns1 - hedge_ratio * returns2)
        
        return strategy_returns.fillna(0)
    
    def backtest_pairs_strategy(self, price1: pd.Series, price2: pd.Series,
                               stock1_name: str, stock2_name: str,
                               entry_threshold: float, exit_threshold: float,
                               stop_loss: float, window: int = 60) -> Dict:
        """
        Backtest a pairs trading strategy.
        
        Returns:
            Dictionary containing backtest results
        """
        try:
            # Ensure we have enough data
            if len(price1) < window + 20 or len(price2) < window + 20:
                raise ValueError("Insufficient data for backtesting")
            
            # Align price series
            aligned_data = pd.concat([price1, price2], axis=1).dropna()
            if len(aligned_data) < window + 20:
                raise ValueError("Insufficient aligned data for backtesting")
            
            p1, p2 = aligned_data.iloc[:, 0], aligned_data.iloc[:, 1]
            
            # Calculate hedge ratio using the first part of the data
            training_period = min(252, len(aligned_data) // 2)  # Use first half or 1 year
            train_p1 = p1.iloc[:training_period]
            train_p2 = p2.iloc[:training_period]
            
            # Simple linear regression to get hedge ratio
            X = train_p2.values.reshape(-1, 1)
            y = train_p1.values
            
            # Calculate hedge ratio manually
            covariance = np.cov(train_p1, train_p2)[0, 1]
            variance_p2 = np.var(train_p2)
            hedge_ratio = covariance / variance_p2 if variance_p2 != 0 else 1.0
            
            # Calculate intercept
            intercept = np.mean(train_p1) - hedge_ratio * np.mean(train_p2)
            
            # Calculate spread for the entire period
            spread = self.calculate_spread(p1, p2, hedge_ratio, intercept)
            
            # Calculate z-score
            zscore = self.calculate_zscore(spread, window)
            
            # Generate trading signals
            signals = self.generate_signals(zscore, entry_threshold, exit_threshold, stop_loss)
            
            # Calculate strategy returns
            strategy_returns = self.calculate_position_returns(p1, p2, signals, hedge_ratio)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # Calculate performance metrics
            metrics = self.calculate_performance_metrics(strategy_returns)
            
            # Identify trades
            trades = self.identify_trades(signals, p1, p2, zscore, hedge_ratio)
            
            # Additional analysis
            spread_stats = {
                'mean': spread.mean(),
                'std': spread.std(),
                'min': spread.min(),
                'max': spread.max(),
                'half_life': self.calculate_half_life(spread)
            }
            
            return {
                'price1': p1,
                'price2': p2,
                'stock1_name': stock1_name,
                'stock2_name': stock2_name,
                'spread': spread,
                'zscore': zscore,
                'signals': signals,
                'strategy_returns': strategy_returns,
                'cumulative_returns': cumulative_returns,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'metrics': metrics,
                'trades': trades,
                'spread_stats': spread_stats
            }
            
        except Exception as e:
            raise ValueError(f"Error in backtesting: {str(e)}")
    
    def calculate_performance_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics for the strategy."""
        # Remove any infinite or NaN values
        clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_returns) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0
            }
        
        # Basic metrics
        total_return = (1 + clean_returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(clean_returns)) - 1
        volatility = clean_returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + clean_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Win rate
        positive_returns = clean_returns[clean_returns > 0]
        win_rate = len(positive_returns) / len(clean_returns) if len(clean_returns) > 0 else 0
        
        # Profit factor
        gross_profit = positive_returns.sum()
        gross_loss = abs(clean_returns[clean_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
    
    def identify_trades(self, signals: pd.Series, price1: pd.Series, price2: pd.Series, 
                       zscore: pd.Series, hedge_ratio: float) -> List[Dict]:
        """Identify individual trades from signals."""
        trades = []
        current_trade = None
        
        for i in range(1, len(signals)):
            prev_signal = signals.iloc[i-1]
            curr_signal = signals.iloc[i]
            
            # Trade entry
            if prev_signal == 0 and curr_signal != 0:
                current_trade = {
                    'entry_date': signals.index[i],
                    'entry_signal': curr_signal,
                    'entry_price1': price1.iloc[i],
                    'entry_price2': price2.iloc[i],
                    'entry_zscore': zscore.iloc[i]
                }
            
            # Trade exit
            elif prev_signal != 0 and curr_signal == 0 and current_trade is not None:
                current_trade.update({
                    'exit_date': signals.index[i],
                    'exit_price1': price1.iloc[i],
                    'exit_price2': price2.iloc[i],
                    'exit_zscore': zscore.iloc[i]
                })
                
                # Calculate trade P&L
                if current_trade['entry_signal'] == 1:  # Long spread
                    pnl = ((current_trade['exit_price1'] - current_trade['entry_price1']) - 
                           hedge_ratio * (current_trade['exit_price2'] - current_trade['entry_price2']))
                else:  # Short spread
                    pnl = -((current_trade['exit_price1'] - current_trade['entry_price1']) - 
                            hedge_ratio * (current_trade['exit_price2'] - current_trade['entry_price2']))
                
                current_trade['pnl'] = pnl
                current_trade['duration_days'] = (current_trade['exit_date'] - current_trade['entry_date']).days
                
                trades.append(current_trade)
                current_trade = None
        
        return trades
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        try:
            # Remove NaN values
            clean_spread = spread.dropna()
            if len(clean_spread) < 10:
                return np.nan
            
            # Calculate first differences
            spread_diff = clean_spread.diff().dropna()
            spread_lag = clean_spread.shift(1).dropna()
            
            # Align the series
            aligned_data = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            if len(aligned_data) < 5:
                return np.nan
            
            y = aligned_data.iloc[:, 0].values
            x = aligned_data.iloc[:, 1].values
            
            # Fit regression: spread_diff = alpha + beta * spread_lag + error
            X = np.column_stack([np.ones(len(x)), x])
            
            try:
                # Use least squares
                coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
                beta = coeffs[1]
                
                # Calculate half-life
                if beta >= 0:
                    return np.nan
                
                half_life = -np.log(2) / beta
                return half_life if half_life > 0 else np.nan
                
            except np.linalg.LinAlgError:
                return np.nan
            
        except Exception:
            return np.nan
