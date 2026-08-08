"""
llm_rl_agent.py  —  LLM-Guided RL Agent
========================================

LLM proposes high-level strategies, RL executes them.
"""

import numpy as np
from typing import Dict, List, Optional


class LLMRLAgent:
    """
    LLM-guided Reinforcement Learning agent.
    
    LLM proposes strategy based on regime, RL executes with reinforcement learning.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.regimes = config.get("regime_detection", ["bull", "bear", "sideways"])
        self.learning_rate = config.get("rl_learning_rate", 0.001)
        self.gamma = config.get("rl_gamma", 0.99)
        self.tau = config.get("rl_tau", 0.005)
        self.exploration_ratio = config.get("exploration_ratio", 0.3)
        
        # Q-values for (regime, action)
        self.q_table = {}
        self.action_space = ["BUY", "HOLD", "SELL"]
        
        # Initialize Q-values
        for regime in self.regimes:
            self.q_table[regime] = np.zeros(len(self.action_space))
        
        # State tracking
        self.current_regime = None
        self.current_action = None
        self.reward_history = []
        
    def detect_regime(self, data: Dict) -> str:
        """Detect market regime."""
        returns = data.get("returns", [])
        if len(returns) < 20:
            return "unknown"
        
        vol = np.std(returns[-20:])
        trend = np.mean(returns[-10:])
        
        if vol > 0.02 and trend > 0.001:
            return "bull"
        elif vol > 0.02 and trend < -0.001:
            return "bear"
        elif vol < 0.01 and abs(trend) < 0.001:
            return "sideways"
        elif vol > 0.02:
            return "volatile"
        else:
            return "calm"
    
    def llm_propose_strategy(self, regime: str, data: Dict) -> Dict:
        """
        LLM-inspired strategy proposal.
        
        In production, this would call an actual LLM API.
        Here, we use rule-based logic that simulates LLM reasoning.
        """
        returns = data.get("returns", [])
        
        strategies = {
            "bull": {
                "bias": "long",
                "risk": "moderate",
                "suggested_action": "BUY",
                "confidence": 0.7,
                "reasoning": "Bull market detected. Momentum is positive. Consider long positions with moderate risk."
            },
            "bear": {
                "bias": "short",
                "risk": "high",
                "suggested_action": "SELL",
                "confidence": 0.7,
                "reasoning": "Bear market detected. Momentum is negative. Consider short positions or cash."
            },
            "sideways": {
                "bias": "neutral",
                "risk": "low",
                "suggested_action": "HOLD",
                "confidence": 0.6,
                "reasoning": "Sideways market detected. Range-bound trading. Wait for breakout."
            },
            "volatile": {
                "bias": "flexible",
                "risk": "high",
                "suggested_action": "HOLD",
                "confidence": 0.5,
                "reasoning": "High volatility detected. Risk is elevated. Consider reducing exposure."
            },
            "calm": {
                "bias": "moderate",
                "risk": "low",
                "suggested_action": "BUY",
                "confidence": 0.6,
                "reasoning": "Calm market detected. Low volatility. Consider moderate long positions."
            }
        }
        
        return strategies.get(regime, {
            "bias": "neutral",
            "risk": "moderate",
            "suggested_action": "HOLD",
            "confidence": 0.4,
            "reasoning": "Unknown regime. Maintain current position."
        })
    
    def select_action(self, data: Dict, explore: bool = True) -> Dict:
        """
        Select action using LLM-guided RL.
        """
        # Detect regime
        regime = self.detect_regime(data)
        self.current_regime = regime
        
        # LLM proposes strategy
        llm_proposal = self.llm_propose_strategy(regime, data)
        
        # Get Q-values for this regime
        q_values = self.q_table.get(regime, np.zeros(len(self.action_space)))
        
        # Exploration-exploitation
        if explore and np.random.random() < self.exploration_ratio:
            # Explore: follow LLM proposal with noise
            action_idx = np.random.choice(len(self.action_space))
        else:
            # Exploit: choose best action from Q-values
            action_idx = np.argmax(q_values)
        
        # Adjust based on LLM proposal
        proposed_action = llm_proposal.get("suggested_action", "HOLD")
        proposed_idx = self.action_space.index(proposed_action) if proposed_action in self.action_space else 1
        
        # Blend LLM and RL: 30% LLM, 70% RL
        if np.random.random() < 0.3:
            action_idx = proposed_idx
        
        self.current_action = action_idx
        
        return {
            "action": self.action_space[action_idx],
            "action_idx": action_idx,
            "regime": regime,
            "llm_proposal": llm_proposal,
            "q_values": q_values.tolist(),
            "exploration": explore
        }
    
    def update(self, reward: float, done: bool = False):
        """
        Update Q-values using RL.
        """
        if self.current_regime is None or self.current_action is None:
            return
        
        q_values = self.q_table[self.current_regime]
        current_q = q_values[self.current_action]
        
        # Q-learning update
        max_next_q = np.max(q_values)
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q * (1 - done) - current_q)
        
        q_values[self.current_action] = new_q
        self.q_table[self.current_regime] = q_values
        
        self.reward_history.append(reward)
