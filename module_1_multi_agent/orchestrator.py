"""
orchestrator.py  —  Multi-Agent Orchestrator
============================================

Coordinates all agents, aggregates signals, and produces final decisions.
"""

import numpy as np
from typing import Dict, List, Optional
from .agents.indicator_agent import IndicatorAgent
from .agents.pattern_agent import PatternAgent
from .agents.trend_agent import TrendAgent
from .agents.risk_agent import RiskAgent
from .agents.sentiment_agent import SentimentAgent


class AgentOrchestrator:
    """Orchestrates multi-agent collaboration."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.agents = []
        self._initialize_agents()
        
        # Voting history
        self.vote_history = []
        self.consensus_history = []
        
    def _initialize_agents(self):
        """Initialize all specialized agents."""
        agent_config = self.config.get("agents", {})
        
        agent_classes = {
            "indicator": IndicatorAgent,
            "pattern": PatternAgent,
            "trend": TrendAgent,
            "risk": RiskAgent,
            "sentiment": SentimentAgent,
        }
        
        for name, config in agent_config.items():
            if name in agent_classes:
                agent = agent_classes[name](config)
                self.agents.append(agent)
    
    def analyze(self, data: Dict) -> Dict:
        """
        Run all agents and aggregate their signals.
        """
        agent_results = []
        
        for agent in self.agents:
            result = agent.analyze(data)
            if result is not None:
                agent_results.append({
                    "agent": agent.name,
                    "signal": result["signal"],
                    "confidence": result["confidence"],
                    "weight": agent.weight,
                    "reasoning": result.get("reasoning", "")
                })
        
        if not agent_results:
            return {
                "signal": 0,
                "consensus": 0,
                "confidence": 0,
                "agents": []
            }
        
        # Weighted average signal
        total_weight = sum(r["weight"] for r in agent_results)
        weighted_signal = sum(r["signal"] * r["weight"] * r["confidence"] for r in agent_results) / (total_weight + 1e-6)
        
        # Consensus (variance of signals)
        signals = [r["signal"] for r in agent_results]
        consensus = 1 - np.std(signals) / 2
        
        # Overall confidence
        avg_confidence = np.mean([r["confidence"] for r in agent_results])
        
        self.vote_history.append(agent_results)
        self.consensus_history.append(consensus)
        
        return {
            "signal": np.clip(weighted_signal, -1, 1),
            "consensus": consensus,
            "confidence": avg_confidence,
            "agents": agent_results
        }
    
    def get_regime(self, data: Dict) -> str:
        """Determine current market regime."""
        # Simple regime detection based on volatility and trend
        returns = data.get("returns", [])
        if len(returns) < 20:
            return "unknown"
        
        vol = np.std(returns[-20:])
        trend = np.mean(returns[-10:])
        
        if vol > 0.02 and trend > 0:
            return "bull_volatile"
        elif vol > 0.02 and trend < 0:
            return "bear_volatile"
        elif vol < 0.01 and trend > 0:
            return "bull_calm"
        elif vol < 0.01 and trend < 0:
            return "bear_calm"
        elif vol > 0.02 and abs(trend) < 0.001:
            return "sideways_volatile"
        else:
            return "sideways_calm"
