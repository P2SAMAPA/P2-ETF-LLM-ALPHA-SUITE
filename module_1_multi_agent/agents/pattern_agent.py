"""
pattern_agent.py  —  Pattern Detection Specialist Agent
========================================================

Detects chart patterns: head & shoulders, flags, triangles, wedges.
"""

import numpy as np
from typing import Dict
from .base_agent import BaseAgent


class PatternAgent(BaseAgent):
    """Chart pattern detection specialist."""
    
    def __init__(self, config: Dict):
        super().__init__("pattern", config)
        self.features = config.get("features", ["head_shoulders", "flag", "triangle", "wedge"])
        
    def analyze(self, data: Dict) -> Dict:
        """Detect chart patterns."""
        prices = data.get("prices", [])
        if len(prices) < 60:
            return {"signal": 0, "confidence": 0.3, "reasoning": "Insufficient data"}
        
        signals = []
        reasons = []
        
        # Head and Shoulders detection (simplified)
        if "head_shoulders" in self.features and len(prices) > 40:
            hs_signal = self._detect_head_shoulders(prices)
            if hs_signal != 0:
                signals.append(hs_signal)
                reasons.append("Head & Shoulders pattern detected")
        
        # Flag pattern detection
        if "flag" in self.features and len(prices) > 30:
            flag_signal = self._detect_flag(prices)
            if flag_signal != 0:
                signals.append(flag_signal)
                reasons.append("Flag pattern detected")
        
        # Triangle pattern detection
        if "triangle" in self.features and len(prices) > 30:
            tri_signal = self._detect_triangle(prices)
            if tri_signal != 0:
                signals.append(tri_signal)
                reasons.append("Triangle pattern detected")
        
        # Wedge pattern detection
        if "wedge" in self.features and len(prices) > 30:
            wedge_signal = self._detect_wedge(prices)
            if wedge_signal != 0:
                signals.append(wedge_signal)
                reasons.append("Wedge pattern detected")
        
        if not signals:
            return {"signal": 0, "confidence": 0.3, "reasoning": "No patterns detected"}
        
        avg_signal = np.mean(signals)
        confidence = min(0.8, 0.4 + 0.1 * len(signals))
        
        return {
            "signal": float(np.clip(avg_signal, -1, 1)),
            "confidence": float(confidence),
            "reasoning": "; ".join(reasons) if reasons else "No clear pattern"
        }
    
    def _detect_head_shoulders(self, prices: np.ndarray) -> float:
        """Detect head and shoulders pattern."""
        if len(prices) < 40:
            return 0
        
        # Find local maxima
        peaks = []
        for i in range(5, len(prices) - 5):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
        
        if len(peaks) < 3:
            return 0
        
        # Check for head and shoulders pattern
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            if p2[1] > p1[1] and p2[1] > p3[1]:
                if abs(p1[1] - p3[1]) / (p1[1] + 1e-6) < 0.05:
                    return -0.7
        
        # Inverse head and shoulders (bullish)
        for i in range(len(peaks) - 2):
            p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
            if p2[1] < p1[1] and p2[1] < p3[1]:
                if abs(p1[1] - p3[1]) / (p1[1] + 1e-6) < 0.05:
                    return 0.7
        
        return 0
    
    def _detect_flag(self, prices: np.ndarray) -> float:
        """Detect flag pattern."""
        if len(prices) < 30:
            return 0
        
        recent = prices[-30:]
        first_third = recent[:10]
        last_two_third = recent[10:]
        
        first_range = np.max(first_third) - np.min(first_third)
        second_range = np.max(last_two_third) - np.min(last_two_third)
        
        if first_range > 0 and second_range / (first_range + 1e-6) < 0.5:
            if first_third[-1] > first_third[0]:
                return 0.6
            else:
                return -0.6
        
        return 0
    
    def _detect_triangle(self, prices: np.ndarray) -> float:
        """Detect triangle pattern."""
        if len(prices) < 30:
            return 0
        
        recent = prices[-30:]
        highs = []
        lows = []
        
        for i in range(5, len(recent) - 5):
            if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                highs.append(recent[i])
            if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                lows.append(recent[i])
        
        if len(highs) < 3 or len(lows) < 3:
            return 0
        
        high_slope = (highs[-1] - highs[0]) / (len(highs) + 1e-6)
        low_slope = (lows[-1] - lows[0]) / (len(lows) + 1e-6)
        
        if high_slope < 0 and low_slope > 0:
            return 0.3
        elif high_slope < 0 and low_slope < 0:
            return -0.5
        elif high_slope > 0 and low_slope > 0:
            return 0.5
        
        return 0
    
    def _detect_wedge(self, prices: np.ndarray) -> float:
        """Detect wedge pattern."""
        if len(prices) < 30:
            return 0
        
        recent = prices[-30:]
        highs = []
        lows = []
        
        for i in range(5, len(recent) - 5):
            if recent[i] > recent[i-1] and recent[i] > recent[i+1]:
                highs.append(recent[i])
            if recent[i] < recent[i-1] and recent[i] < recent[i+1]:
                lows.append(recent[i])
        
        if len(highs) < 2 or len(lows) < 2:
            return 0
        
        high_slope = (highs[-1] - highs[0]) / (len(highs) + 1e-6)
        low_slope = (lows[-1] - lows[0]) / (len(lows) + 1e-6)
        
        if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
            return -0.6
        elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            return 0.6
        
        return 0
