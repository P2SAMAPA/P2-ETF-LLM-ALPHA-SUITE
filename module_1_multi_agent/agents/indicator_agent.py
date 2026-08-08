"""
indicator_agent.py  —  Indicator Specialist Agent
==================================================

Analyzes technical indicators: RSI, MACD, Bollinger Bands.
"""

import numpy as np
from typing import Dict
from .base_agent import BaseAgent


class IndicatorAgent(BaseAgent):
    """Technical indicator specialist."""
    
    def __init__(self, config: Dict):
        super().__init__("indicator", config)
        self.features = config.get("features", ["rsi", "macd", "bb", "volume"])
        
    def analyze(self, data: Dict) -> Dict:
        """Analyze technical indicators."""
        prices = data.get("prices", [])
        if len(prices) < 60:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        returns = np.log(prices / np.roll(prices, 1))[1:]
        
        signals = []
        confidences = []
        reasons = []
        
        # RSI
        if "rsi" in self.features and len(prices) > 14:
            rsi = self._compute_rsi(returns, 14)
            if rsi < 30:
                signals.append(0.7)
                confidences.append(0.7)
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > 70:
                signals.append(-0.7)
                confidences.append(0.7)
                reasons.append(f"RSI overbought ({rsi:.1f})")
            else:
                signals.append(0)
                confidences.append(0.3)
                reasons.append(f"RSI neutral ({rsi:.1f})")
        
        # MACD
        if "macd" in self.features and len(returns) > 26:
            macd_signal = self._compute_macd(returns)
            if macd_signal > 0:
                signals.append(0.5)
                confidences.append(0.6)
                reasons.append("MACD bullish crossover")
            else:
                signals.append(-0.5)
                confidences.append(0.6)
                reasons.append("MACD bearish crossover")
        
        # Bollinger Bands
        if "bb" in self.features and len(prices) > 20:
            bb_signal = self._compute_bollinger(prices)
            if bb_signal > 0:
                signals.append(0.6)
                confidences.append(0.5)
                reasons.append("Price below lower Bollinger band")
            elif bb_signal < 0:
                signals.append(-0.6)
                confidences.append(0.5)
                reasons.append("Price above upper Bollinger band")
            else:
                signals.append(0)
                confidences.append(0.3)
                reasons.append("Price within Bollinger bands")
        
        if not signals:
            return {"signal": 0, "confidence": 0.3, "reasoning": "No clear indicators"}
        
        # Aggregate
        avg_signal = np.mean(signals)
        avg_confidence = np.mean(confidences)
        
        self.signal_history.append(avg_signal)
        self.confidence_history.append(avg_confidence)
        
        return {
            "signal": float(avg_signal),
            "confidence": float(avg_confidence),
            "reasoning": "; ".join(reasons[:3]) if reasons else "No clear signal"
        }
    
    def _compute_rsi(self, returns: np.ndarray, period: int = 14) -> float:
        """Compute RSI."""
        if len(returns) < period:
            return 50
        
        gains = returns[returns > 0]
        losses = -returns[returns < 0]
        
        avg_gain = np.mean(gains[-period:]) if len(gains) > period else np.mean(gains)
        avg_loss = np.mean(losses[-period:]) if len(losses) > period else np.mean(losses)
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def _compute_macd(self, returns: np.ndarray) -> float:
        """Compute MACD signal."""
        if len(returns) < 26:
            return 0
        
        # Simple MACD: 12-day EMA - 26-day EMA
        ema12 = self._ema(returns, 12)
        ema26 = self._ema(returns, 26)
        macd = ema12 - ema26
        
        # Signal line: 9-day EMA of MACD
        signal = self._ema(macd, 9)
        
        if len(macd) > 0 and len(signal) > 0:
            return float(macd[-1] - signal[-1])
        return 0
    
    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute EMA."""
        if len(data) < period:
            return np.array([0])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _compute_bollinger(self, prices: np.ndarray) -> float:
        """Compute Bollinger Bands signal."""
        if len(prices) < 20:
            return 0
        
        # Simple moving average
        ma = np.mean(prices[-20:])
        std = np.std(prices[-20:])
        
        upper = ma + 2 * std
        lower = ma - 2 * std
        current = prices[-1]
        
        if current < lower:
            return 1.0  # Buy signal (oversold)
        elif current > upper:
            return -1.0  # Sell signal (overbought)
        else:
            return 0.0  # Neutral
