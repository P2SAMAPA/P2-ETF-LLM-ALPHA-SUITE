"""
alpha_generator.py  —  Symbolic Alpha Generator
================================================

Uses LLM-inspired rule generation to discover trading signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore")


class SymbolicAlphaGenerator:
    """Generates and evaluates symbolic trading rules."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.n_alphas = config.get("n_alphas", 100)
        self.backtest_window = config.get("backtest_window", 252)
        self.evaluation_metric = config.get("evaluation_metric", "sharpe")
        self.complexity_penalty = config.get("complexity_penalty", 0.01)
        
        self.alpha_pool = []
        self.top_alphas = []
        
    def generate_alphas(self, data: Dict) -> List[Dict]:
        """
        Generate candidate alphas using rule templates.
        """
        prices = data.get("prices", [])
        returns = np.log(prices / np.roll(prices, 1))[1:]
        
        if len(returns) < 100:
            return []
        
        alphas = []
        
        # Template 1: Moving Average Crossover
        for fast in [5, 10, 20]:
            for slow in [20, 30, 50]:
                if fast < slow:
                    alpha = self._ma_crossover(returns, fast, slow)
                    alphas.append({
                        "name": f"MA_CROSS_{fast}_{slow}",
                        "rule": f"MA{fast} > MA{slow}",
                        "signal": alpha,
                        "params": {"fast": fast, "slow": slow}
                    })
        
        # Template 2: Momentum
        for period in [10, 20, 30, 60]:
            alpha = self._momentum(returns, period)
            alphas.append({
                "name": f"MOM_{period}",
                "rule": f"Return over {period} days",
                "signal": alpha,
                "params": {"period": period}
            })
        
        # Template 3: Mean Reversion
        for period in [5, 10, 20]:
            alpha = self._mean_reversion(returns, period)
            alphas.append({
                "name": f"MR_{period}",
                "rule": f"Mean reversion over {period} days",
                "signal": alpha,
                "params": {"period": period}
            })
        
        # Template 4: Volatility
        for period in [10, 20, 30]:
            alpha = self._volatility(returns, period)
            alphas.append({
                "name": f"VOL_{period}",
                "rule": f"Volatility over {period} days",
                "signal": alpha,
                "params": {"period": period}
            })
        
        # Template 5: RSI-like
        for period in [7, 14, 21]:
            alpha = self._rsi_like(returns, period)
            alphas.append({
                "name": f"RSI_{period}",
                "rule": f"RSI over {period} days",
                "signal": alpha,
                "params": {"period": period}
            })
        
        self.alpha_pool = alphas
        return alphas
    
    def evaluate_alphas(self, alphas: List[Dict], returns: np.ndarray) -> List[Dict]:
        """
        Evaluate and rank alphas.
        """
        evaluated = []
        
        for alpha in alphas:
            signal = alpha["signal"]
            
            if len(signal) < 20:
                continue
            
            # Sharpe ratio of the signal
            sharpe = np.mean(signal) / (np.std(signal) + 1e-6)
            
            # Correlation with future returns (basic alpha effectiveness)
            future_returns = returns[1:]
            if len(signal) > len(future_returns):
                signal = signal[:len(future_returns)]
            elif len(signal) < len(future_returns):
                future_returns = future_returns[:len(signal)]
            
            corr = np.corrcoef(signal, future_returns)[0, 1] if len(signal) > 1 else 0
            
            # Complexity penalty
            complexity = len(str(alpha["rule"])) / 50
            complexity_score = 1 - self.complexity_penalty * complexity
            
            # Combined score
            if self.evaluation_metric == "sharpe":
                score = sharpe * complexity_score
            else:
                score = corr * complexity_score
            
            evaluated.append({
                **alpha,
                "sharpe": sharpe,
                "correlation": corr,
                "score": score,
                "complexity": complexity,
                "complexity_score": complexity_score
            })
        
        # Sort by score
        evaluated = sorted(evaluated, key=lambda x: x["score"], reverse=True)
        
        self.top_alphas = evaluated[:self.config.get("selection_top_n", 10)]
        return evaluated
    
    def _ma_crossover(self, returns: np.ndarray, fast: int, slow: int) -> np.ndarray:
        """Moving average crossover signal."""
        if len(returns) < slow:
            return np.zeros(len(returns))
        
        ma_fast = np.convolve(returns, np.ones(fast)/fast, mode='valid')
        ma_slow = np.convolve(returns, np.ones(slow)/slow, mode='valid')
        
        # Align lengths
        min_len = min(len(ma_fast), len(ma_slow))
        ma_fast = ma_fast[-min_len:]
        ma_slow = ma_slow[-min_len:]
        
        # Signal: 1 if fast > slow, -1 if fast < slow
        signal = np.zeros(min_len)
        signal[ma_fast > ma_slow] = 1
        signal[ma_fast < ma_slow] = -1
        
        # Pad to original length
        padded = np.zeros(len(returns))
        padded[-min_len:] = signal
        
        return padded
    
    def _momentum(self, returns: np.ndarray, period: int) -> np.ndarray:
        """Momentum signal."""
        if len(returns) < period:
            return np.zeros(len(returns))
        
        momentum = np.zeros(len(returns))
        for i in range(period, len(returns)):
            momentum[i] = np.sum(returns[i-period:i])
        
        # Normalize
        momentum = momentum / (np.std(momentum) + 1e-6)
        return np.clip(momentum, -1, 1)
    
    def _mean_reversion(self, returns: np.ndarray, period: int) -> np.ndarray:
        """Mean reversion signal."""
        if len(returns) < period:
            return np.zeros(len(returns))
        
        mean_rev = np.zeros(len(returns))
        for i in range(period, len(returns)):
            mean_ret = np.mean(returns[i-period:i])
            mean_rev[i] = -mean_ret
        
        # Normalize
        mean_rev = mean_rev / (np.std(mean_rev) + 1e-6)
        return np.clip(mean_rev, -1, 1)
    
    def _volatility(self, returns: np.ndarray, period: int) -> np.ndarray:
        """Volatility signal."""
        if len(returns) < period:
            return np.zeros(len(returns))
        
        vol = np.zeros(len(returns))
        for i in range(period, len(returns)):
            vol[i] = np.std(returns[i-period:i])
        
        # Inverse volatility (low vol = buy signal)
        vol = -vol
        vol = vol / (np.std(vol) + 1e-6)
        return np.clip(vol, -1, 1)
    
    def _rsi_like(self, returns: np.ndarray, period: int) -> np.ndarray:
        """RSI-like signal."""
        if len(returns) < period:
            return np.zeros(len(returns))
        
        rsi = np.zeros(len(returns))
        for i in range(period, len(returns)):
            window = returns[i-period:i]
            gains = window[window > 0].sum()
            losses = -window[window < 0].sum()
            
            if losses == 0:
                rsi[i] = 100
            else:
                rs = gains / losses
                rsi[i] = 100 - (100 / (1 + rs))
        
        # Normalize to [-1, 1]
        rsi = (rsi - 50) / 50
        return np.clip(rsi, -1, 1)
