"""
risk_agent.py  —  Risk Assessment Specialist Agent
===================================================

Assesses risk: volatility, drawdown, VaR, tail risk.
"""

import numpy as np
from typing import Dict
from .base_agent import BaseAgent


class RiskAgent(BaseAgent):
    """Risk assessment specialist."""
    
    def __init__(self, config: Dict):
        super().__init__("risk", config)
        self.features = config.get("features", ["volatility", "drawdown", "var", "tail_risk"])
        
    def analyze(self, data: Dict) -> Dict:
        """Assess risk."""
        prices = data.get("prices", [])
        if len(prices) < 60:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        returns = np.log(prices / np.roll(prices, 1))[1:]
        
        signals = []
        confidences = []
        reasons = []
        
        # Volatility
        if "volatility" in self.features:
            vol = np.std(returns[-20:])
            vol_signal = -np.clip(vol * 50, -1, 1)
            signals.append(vol_signal)
            confidences.append(0.5 + 0.3 * abs(vol_signal))
            reasons.append(f"Volatility: {vol:.4f}")
        
        # Drawdown
        if "drawdown" in self.features and len(prices) > 20:
            cum_returns = np.cumsum(returns[-60:])
            running_max = np.maximum.accumulate(cum_returns)
            drawdown = running_max - cum_returns
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
            
            dd_signal = -np.clip(max_dd * 5, -1, 1)
            signals.append(dd_signal)
            confidences.append(0.5 + 0.3 * abs(dd_signal))
            reasons.append(f"Max Drawdown: {max_dd:.4f}")
        
        # VaR
        if "var" in self.features and len(returns) > 20:
            var_95 = np.percentile(returns[-60:], 5)
            var_signal = -np.clip(var_95 * 100, -1, 1)
            signals.append(var_signal)
            confidences.append(0.5 + 0.3 * abs(var_signal))
            reasons.append(f"VaR (95%): {var_95:.4f}")
        
        # Tail Risk (kurtosis)
        if "tail_risk" in self.features and len(returns) > 20:
            # Use scipy if available, otherwise simple calculation
            try:
                from scipy.stats import kurtosis
                kurt = kurtosis(returns[-60:])
            except:
                mean = np.mean(returns[-60:])
                std = np.std(returns[-60:])
                kurt = np.mean(((returns[-60:] - mean) / (std + 1e-6)) ** 4) - 3
            
            tail_signal = -np.clip(kurt / 5, -1, 1)
            signals.append(tail_signal)
            confidences.append(0.5 + 0.3 * abs(tail_signal))
            reasons.append(f"Kurtosis: {kurt:.2f}")
        
        if not signals:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        avg_signal = np.mean(signals)
        avg_confidence = np.mean(confidences)
        
        return {
            "signal": float(np.clip(avg_signal, -1, 1)),
            "confidence": float(avg_confidence),
            "reasoning": "; ".join(reasons) if reasons else "No clear risk signal"
        }
