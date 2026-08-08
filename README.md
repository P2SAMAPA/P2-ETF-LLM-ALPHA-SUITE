# P2-LLM-ALPHA-SUITE

**LLM / Multi-Agent Systems for Alpha & Strategy Discovery**

Part of the **P2Quant Engine Suite** · P2SAMAPA

---

## What This Suite Does

This is an integrated suite of 4 interconnected modules that work together to discover, evaluate, and execute alpha strategies:

| Module | What it does | When to use |
|--------|--------------|-------------|
| **1. Multi-Agent Alpha Discovery** | Specialized agents collaborate to discover trading signals | Daily alpha generation |
| **2. Symbolic Alpha Generator** | LLM-inspired rule generation and backtesting | Strategy discovery |
| **3. LLM-Guided RL Agent** | High-level strategy planning + RL execution | Regime-based trading |
| **4. Multi-Modal Evaluator** | Combines signals with dynamic weighting | Portfolio construction |

---

## Architecture
┌─────────────────────────────────────────────────────────────────────────────┐
│ P2-LLM-ALPHA-SUITE │
├─────────────────────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ ORCHESTRATOR (Master Agent) │ │
│ │ Coordinates all modules & decisions │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ┌──────────────────────────┼──────────────────────────┐ │
│ ▼ ▼ ▼ │
│ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ Agent 1 │ │ Agent 2 │ │ Agent 3 │ │
│ │ Indicator │ │ Pattern │ │ Trend │ │
│ │ Specialist │ │ Detector │ │ Analyst │ │
│ └──────────────┘ └──────────────┘ └──────────────┘ │
│ │ │ │ │
│ ▼ ▼ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ SYMBOLIC ALPHA GENERATOR │ │
│ │ LLM-inspired → backtests → evaluates → selects │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ LLM-GUIDED RL AGENT │ │
│ │ Regime detection → Strategy proposal → RL execution │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│ │ │
│ ▼ │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ MULTI-MODAL EVALUATOR │ │
│ │ Weighted combination → Portfolio construction │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘

text

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-LLM-ALPHA-SUITE
cd P2-LLM-ALPHA-SUITE
pip install -r requirements.txt

export HF_TOKEN=hf_...
python trainer.py

streamlit run streamlit_app.py
GitHub Actions
Runs automatically at 00:30 UTC Monday–Saturday.

Required secret: HF_TOKEN

References
Zhang, J., et al. (2024). QuantAgent: A Multi-Agent System for Alpha Discovery. arXiv.

Li, X., et al. (2024). AlphaQuanter: LLM-driven Quantitative Trading. arXiv.

Liu, Z., et al. (2024). R&D-Agent-Quant: Research & Development Agent for Quantitative Trading. arXiv.

Wang, L., et al. (2024). Logic-Q: Program-sketch Guidance for Quantitative Trading. ICLR 2024.

text

---

## Complete File Structure
P2-LLM-ALPHA-SUITE/
├── README.md ✅ Complete
├── config.py ✅ Complete
├── data_manager.py ✅ Complete
├── trainer.py ✅ Complete
├── push_results.py ✅ Complete
├── streamlit_app.py ✅ Complete
├── us_calendar.py ✅ Complete
├── requirements.txt ✅ Complete
├── module_1_multi_agent/
│ ├── init.py ✅ Complete
│ ├── orchestrator.py ✅ Complete
│ └── agents/
│ ├── init.py ✅ Complete
│ ├── base_agent.py ✅ Complete
│ ├── indicator_agent.py ✅ Complete
│ ├── pattern_agent.py ✅ Complete
│ ├── trend_agent.py ✅ Complete
│ ├── risk_agent.py ✅ Complete
│ └── sentiment_agent.py ✅ Complete
├── module_2_symbolic_alpha/
│ ├── init.py
│ └── alpha_generator.py ✅ Complete
├── module_3_llm_rl/
│ ├── init.py
│ └── llm_rl_agent.py ✅ Complete
├── module_4_evaluator/
│ ├── init.py
│ └── signal_combiner.py ✅ Complete
└── .github/
└── workflows/
└── daily.yml ✅ Complete# P2-ETF-LLM-ALPHA-SUITE
