"""
module_1_multi_agent  —  Multi-Agent Alpha Discovery Module
============================================================

Specialized agents collaborate to discover trading signals.
"""

from .orchestrator import AgentOrchestrator
from .agents.base_agent import BaseAgent
from .agents.indicator_agent import IndicatorAgent
from .agents.pattern_agent import PatternAgent
from .agents.trend_agent import TrendAgent
from .agents.risk_agent import RiskAgent
from .agents.sentiment_agent import SentimentAgent

__all__ = [
    'AgentOrchestrator',
    'BaseAgent',
    'IndicatorAgent',
    'PatternAgent',
    'TrendAgent',
    'RiskAgent',
    'SentimentAgent'
]
