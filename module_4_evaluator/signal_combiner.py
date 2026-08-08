"""
signal_combiner.py  —  Multi-Modal Signal Combiner
===================================================

Combines signals from all modules with dynamic weighting.
"""

import numpy as np
from typing import Dict, List


class SignalCombiner:
    """Combines signals from multiple sources with adaptive weights."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.weights = {}
        self.weight_history = []
        self.performance_history = []
        
    def update_weights(self, signals: Dict, performance: Dict):
        """
        Update weights based on recent performance.
        """
        for name, perf in performance.items():
            # Increase weight for good performers
            if name in self.weights:
                self.weights[name] = self.weights[name] * (1 + 0.1 * perf)
                self.weights[name] = np.clip(self.weights[name], 0.05, 0.50)
            else:
                self.weights[name] = 1.0 / len(signals)
        
        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            for name in self.weights:
                self.weights[name] = self.weights[name] / total
        
        self.weight_history.append(self.weights.copy())
    
    def combine(self, signals: Dict) -> Dict:
        """
        Combine signals with current weights.
        """
        if not signals:
            return {"signal": 0, "confidence": 0, "details": {}}
        
        # Initialize weights if empty
        if not self.weights:
            for name in signals.keys():
                self.weights[name] = 1.0 / len(signals)
        
        weighted_signal = 0
        total_weight = 0
        details = {}
        
        for name, value in signals.items():
            weight = self.weights.get(name, 0.1)
            weighted_signal += value["signal"] * weight
            total_weight += weight
            details[name] = {
                "signal": value["signal"],
                "weight": weight,
                "confidence": value.get("confidence", 0.5)
            }
        
        if total_weight > 0:
            weighted_signal = weighted_signal / total_weight
        
        confidence = np.mean([v.get("confidence", 0.5) for v in signals.values()])
        
        return {
            "signal": np.clip(weighted_signal, -1, 1),
            "confidence": confidence,
            "details": details
        }
