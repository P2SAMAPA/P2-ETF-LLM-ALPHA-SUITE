"""
sentiment_agent.py  —  Sentiment Analysis Specialist Agent
===========================================================

Analyzes market sentiment: put/call ratio, fear/greed, volatility skew.
"""

import numpy as np
from .base_agent import BaseAgent


class SentimentAgent(BaseAgent):
    """Sentiment analysis specialist."""
    
    def __init__(self, config: Dict):
        super().__init__("sentiment", config)
        self.features = config.get("features", ["put_call", "fear_greed", "volatility_skew"])
        
    def analyze(self, data: Dict) -> Dict:
        """Analyze market sentiment."""
        prices = data.get("prices", [])
        returns = data.get("returns", [])
        
        if len(prices) < 20:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        signals = []
        confidences = []
        reasons = []
        
        # Put/Call ratio (simulated using volatility)
        if "put_call" in self.features and len(returns) > 20:
            vol = np.std(returns[-20:])
            # High volatility = high put/call = bearish
            pc_signal = -np.clip((vol - 0.01) * 50, -1, 1)
            signals.append(pc_signal)
            confidences.append(0.5 + 0.3 * abs(pc_signal))
            reasons.append(f"Volatility: {vol:.4f}")
        
        # Fear/Greed (using recent performance)
        if "fear_greed" in self.features and len(returns) > 20:
            recent_return = np.mean(returns[-10:]) * 100
            # Positive returns = greed = overbought (sell signal)
            fg_signal = -np.clip(recent_return / 5, -1, 1)
            signals.append(fg_signal)
            confidences.append(0.5 + 0.3 * abs(fg_signal))
            reasons.append(f"Recent return: {recent_return:.2f}%")
        
        # Volatility skew (using return distribution)
        if "volatility_skew" in self.features and len(returns) > 20:
            skew = pd.Series(returns[-60:]).skew() if len(returns) >= 60 else 0
            # Negative skew = downside risk = bearish
            skew_signal = -np.clip(skew, -1, 1)
            signals.append(skew_signal)
            confidences.append(0.5 + 0.3 * abs(skew_signal))
            reasons.append(f"Skew: {skew:.2f}")
        
        if not signals:
            return {"signal": 0, "confidence": 0.3, "reasoning": "No clear sentiment"}
        
        avg_signal = np.mean(signals)
        avg_confidence = np.mean(confidences)
        
        return {
            "signal": np.clip(avg_signal, -1, 1),
            "confidence": avg_confidence,
            "reasoning": "; ".join(reasons) if reasons else "No clear sentiment"
        }
