"""
base_agent.py  —  Base Agent Class
===================================

Defines the interface for all specialized agents.
"""

import numpy as np
from typing import Dict, List, Optional


class BaseAgent:
    """Base class for all specialized agents."""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.weight = config.get("weight", 0.1)
        self.confidence_history = []
        self.signal_history = []
        
    def analyze(self, data: Dict) -> Dict:
        """
        Analyze market data and return signals.
        
        Returns:
            signal: float between -1 (sell) and 1 (buy)
            confidence: float between 0 and 1
            reasoning: str (explanation)
        """
        raise NotImplementedError
    
    def update_weight(self, performance: float):
        """Update agent weight based on performance."""
        # Simple performance-based weight update
        self.weight = self.weight * (1 + 0.1 * performance)
        self.weight = np.clip(self.weight, 0.05, 0.50)
        
    def get_confidence(self) -> float:
        """Get average confidence over recent history."""
        if len(self.confidence_history) == 0:
            return 0.5
        return np.mean(self.confidence_history[-20:])
    
    def reset(self):
        """Reset agent state."""
        self.confidence_history = []
        self.signal_history = []
