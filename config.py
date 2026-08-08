"""
config.py  —  Main Configuration for P2-LLM-ALPHA-SUITE
========================================================

Defines:
  - UNIVERSES: ETF ticker sets
  - AGENTS: Agent roles and configurations
  - SYMBOLIC_ALPHA: Alpha generation parameters
  - LLM_RL: LLM-guided RL parameters
  - EVALUATOR: Multi-modal evaluation parameters
  - WINDOWS: Time windows for analysis
"""

# ── HuggingFace ──────────────────────────────────────────────────────────────

HF_TOKEN = ""
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
RESULTS_REPO = "P2SAMAPA/p2-llm-alpha-suite-results"


# ── ETF Universes ────────────────────────────────────────────────────────────

UNIVERSES = {
    "FI_COMMODITIES": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
    ],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI",
        "XLY", "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "SOXX", "SMH", "URA",
        "XBI", "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}


# ── Windows ──────────────────────────────────────────────────────────────────

WINDOWS = [63, 126, 252, 504]
PRIMARY_WINDOW = 252


# ── Multi-Agent Configuration ──────────────────────────────────────────────

AGENTS = {
    "indicator": {
        "weight": 0.20,
        "description": "Technical indicator specialist (RSI, MACD, Bollinger)",
        "features": ["rsi", "macd", "bb", "volume"]
    },
    "pattern": {
        "weight": 0.20,
        "description": "Chart pattern detector (head & shoulders, flags, triangles)",
        "features": ["head_shoulders", "flag", "triangle", "wedge"]
    },
    "trend": {
        "weight": 0.25,
        "description": "Trend analysis (momentum, moving averages, trendlines)",
        "features": ["momentum", "ma_cross", "trend_strength"]
    },
    "risk": {
        "weight": 0.15,
        "description": "Risk assessment (volatility, drawdown, tail risk)",
        "features": ["volatility", "drawdown", "var", "tail_risk"]
    },
    "sentiment": {
        "weight": 0.10,
        "description": "Market sentiment (put/call ratio, fear/greed)",
        "features": ["put_call", "fear_greed", "volatility_skew"]
    },
    "macro": {
        "weight": 0.10,
        "description": "Macro analysis (economic indicators, central bank)",
        "features": ["gdp", "inflation", "rates", "credit_spread"]
    }
}


# ── Symbolic Alpha Parameters ──────────────────────────────────────────────

SYMBOLIC_ALPHA = {
    "n_alphas": 100,            # Number of alphas to generate
    "backtest_window": 252,     # Backtest window
    "evaluation_metric": "sharpe",  # or "sortino", "drawdown"
    "selection_top_n": 10,      # Top N alphas to select
    "complexity_penalty": 0.01,  # Penalty for complex rules
    "min_observations": 50,     # Minimum observations for validity
}


# ── LLM-Guided RL Parameters ──────────────────────────────────────────────

LLM_RL = {
    "regime_detection": ["bull", "bear", "sideways", "volatile", "calm"],
    "strategy_proposal": "llm",  # llm or rule-based
    "rl_learning_rate": 0.001,
    "rl_gamma": 0.99,
    "rl_tau": 0.005,
    "exploration_ratio": 0.3,
}


# ── Multi-Modal Evaluator ──────────────────────────────────────────────────

EVALUATOR = {
    "weight_adaptation": "dynamic",  # dynamic or fixed
    "lookback_weight": 63,      # Lookback for weight adaptation
    "performance_metric": "sharpe",
    "risk_target": 0.15,        # Annual volatility target
    "max_position": 0.20,       # Max position per ETF
    "rebalance_frequency": 21,  # Days between rebalancing
}


# ── Macro Signals ────────────────────────────────────────────────────────────

MACRO_SIGNALS = [
    ("VIX",       "VIX",           0.30, -1.0),
    ("T10Y2Y",    "10Y–2Y Spread", 0.25, +1.0),
    ("DXY",       "DXY",           0.20, -1.0),
    ("IG_SPREAD", "IG Spread",     0.15, -1.0),
    ("HY_SPREAD", "HY Spread",     0.10, -1.0),
]

MACRO_COLS_CORE = ["VIX", "T10Y2Y", "DXY"]
MACRO_COLS_EXTENDED = ["IG_SPREAD", "HY_SPREAD"]
