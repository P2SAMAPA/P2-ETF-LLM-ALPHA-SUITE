"""
trend_agent.py  —  Trend Analysis Specialist Agent
===================================================

Analyzes trends: momentum, moving averages, trend strength.
"""

import numpy as np
from .base_agent import BaseAgent


class TrendAgent(BaseAgent):
    """Trend analysis specialist."""
    
    def __init__(self, config: Dict):
        super().__init__("trend", config)
        self.features = config.get("features", ["momentum", "ma_cross", "trend_strength"])
        
    def analyze(self, data: Dict) -> Dict:
        """Analyze trends."""
        prices = data.get("prices", [])
        if len(prices) < 60:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        returns = np.log(prices / np.roll(prices, 1))[1:]
        
        signals = []
        confidences = []
        reasons = []
        
        # Momentum
        if "momentum" in self.features:
            momentum = np.mean(returns[-20:])
            mom_signal = np.clip(momentum * 100, -1, 1)
            signals.append(mom_signal)
            confidences.append(0.6 + 0.3 * abs(mom_signal))
            reasons.append(f"Momentum: {momentum:.4f}")
        
        # Moving Average Crossover
        if "ma_cross" in self.features and len(prices) > 50:
            ma_fast = np.mean(prices[-20:])
            ma_slow = np.mean(prices[-50:])
            
            if ma_fast > ma_slow:
                signals.append(0.5)
                confidences.append(0.6)
                reasons.append("MA20 > MA50 (bullish)")
            else:
                signals.append(-0.5)
                confidences.append(0.6)
                reasons.append("MA20 < MA50 (bearish)")
        
        # Trend Strength (using ADX-like calculation)
        if "trend_strength" in self.features and len(returns) > 14:
            strength = self._compute_trend_strength(returns)
            if strength > 0.3:
                signals.append(np.sign(np.mean(returns[-20:])) * strength)
                confidences.append(0.5 + 0.3 * strength)
                reasons.append(f"Trend strength: {strength:.2f}")
        
        if not signals:
            return {"signal": 0, "confidence": 0.3, "reasoning": "No clear trend"}
        
        avg_signal = np.mean(signals)
        avg_confidence = np.mean(confidences)
        
        return {
            "signal": np.clip(avg_signal, -1, 1),
            "confidence": avg_confidence,
            "reasoning": "; ".join(reasons) if reasons else "No clear trend"
        }
    
    def _compute_trend_strength(self, returns: np.ndarray) -> float:
        """Compute trend strength (ADX-like)."""
        if len(returns) < 14:
            return 0
        
        # Directional movement
        pos_movement = np.maximum(0, returns)
        neg_movement = np.maximum(0, -returns)
        
        # Average directional movement
        atr = np.mean(np.abs(returns[-14:]))
        
        if atr == 0:
            return 0
        
        di_pos = np.mean(pos_movement[-14:]) / atr
        di_neg = np.mean(neg_movement[-14:]) / atr
        
        # Directional index
        dx = abs(di_pos - di_neg) / (di_pos + di_neg + 1e-6)
        adx = dx  # Simplified
        
        return adx
