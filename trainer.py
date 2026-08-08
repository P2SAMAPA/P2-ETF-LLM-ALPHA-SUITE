"""
trainer.py  —  Main Orchestrator for P2-LLM-ALPHA-SUITE
========================================================

Coordinates all modules to produce final trading signals.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional
import numpy as np
import pandas as pd
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data_manager import load_master_data, validate_data
from push_results import upload_results

# Import modules
from module_1_multi_agent.orchestrator import AgentOrchestrator
from module_2_symbolic_alpha.alpha_generator import SymbolicAlphaGenerator
from module_3_llm_rl.llm_rl_agent import LLMRLAgent
from module_4_evaluator.signal_combiner import SignalCombiner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def safe_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def get_action(z_score: float) -> str:
    if z_score > 0.15:
        return "STRONG BUY"
    elif z_score > 0.05:
        return "BUY"
    elif z_score > -0.05:
        return "HOLD"
    elif z_score > -0.15:
        return "REDUCE"
    else:
        return "STRONG SELL"


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Run the full LLM-Alpha-Suite pipeline."""
    token = hf_token or config.HF_TOKEN or os.environ.get("HF_TOKEN")
    if not token:
        logger.warning("HF_TOKEN not set — will skip HuggingFace upload.")

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("🔄 Loading master data from HuggingFace...")
    try:
        prices_df, macro_df = load_master_data(token)
        validate_data(prices_df, macro_df)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"✅ Loaded {len(prices_df)} days, {len(prices_df.columns)} ETFs")

    run_date = datetime.now().strftime("%Y-%m-%d")

    # ── Results containers ────────────────────────────────────────────────────
    results_tab1 = {"run_date": run_date, "universes": {}}
    results_tab2 = {"run_date": run_date, "universes": {}}

    # ── Process each universe ────────────────────────────────────────────────
    for universe_name, tickers in config.UNIVERSES.items():
        logger.info(f"\n🧠 Processing universe: {universe_name}")

        available = [t for t in tickers if t in prices_df.columns]
        logger.info(f"   Available: {len(available)}/{len(tickers)}")

        if not available:
            continue

        # ── Initialize modules ──────────────────────────────────────────────
        agent_orchestrator = AgentOrchestrator(config.__dict__)
        alpha_generator = SymbolicAlphaGenerator(config.SYMBOLIC_ALPHA)
        llm_rl_agent = LLMRLAgent(config.LLM_RL)
        signal_combiner = SignalCombiner(config.EVALUATOR)

        ticker_scores = {}

        for ticker in available:
            logger.info(f"   Computing {ticker}...")
            prices = prices_df[ticker].values
            returns = np.log(prices / np.roll(prices, 1))[1:]

            data = {
                "prices": prices,
                "returns": returns,
                "ticker": ticker
            }

            # ── Module 1: Multi-Agent Alpha Discovery ──────────────────────
            agent_result = agent_orchestrator.analyze(data)
            agent_signal = agent_result.get("signal", 0)

            # ── Module 2: Symbolic Alpha ──────────────────────────────────────
            alphas = alpha_generator.generate_alphas(data)
            evaluated = alpha_generator.evaluate_alphas(alphas, returns) if alphas else []
            alpha_signal = evaluated[0]["signal"][-1] if evaluated else 0

            # ── Module 3: LLM-Guided RL ──────────────────────────────────────
            rl_result = llm_rl_agent.select_action(data)
            rl_signal = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}.get(rl_result["action"], 0.0)

            # ── Module 4: Signal Combination ──────────────────────────────────
            signals = {
                "multi_agent": {"signal": agent_signal, "confidence": agent_result.get("confidence", 0.5)},
                "symbolic_alpha": {"signal": alpha_signal, "confidence": 0.5},
                "llm_rl": {"signal": rl_signal, "confidence": rl_result.get("llm_proposal", {}).get("confidence", 0.5)}
            }
            
            combined = signal_combiner.combine(signals)
            final_signal = combined["signal"]

            # ── Store results ──────────────────────────────────────────────
            z_score = final_signal * 5
            ticker_scores[ticker] = z_score

        # ── Normalize z-scores ──────────────────────────────────────────────
        z_values = np.array(list(ticker_scores.values()))
        if len(z_values) > 1 and np.std(z_values) > 1e-6:
            mean_z = np.mean(z_values)
            std_z = np.std(z_values)
            for ticker in ticker_scores:
                ticker_scores[ticker] = (ticker_scores[ticker] - mean_z) / std_z

        # ── Rank and build results ──────────────────────────────────────────
        ranked = sorted(ticker_scores.items(), key=lambda x: x[1], reverse=True)
        
        top_buys = [{"ticker": t, "z_score": z} for t, z in ranked[:5]]
        top_sells = [{"ticker": t, "z_score": z} for t, z in ranked[-5:]]

        results_tab1["universes"][universe_name] = {
            "top_buys": top_buys,
            "top_sells": top_sells,
            "full_scores": {
                t: {"z_score": z, "action": get_action(z)}
                for t, z in ticker_scores.items()
            }
        }

        # ── Tab 2: Module breakdown ──────────────────────────────────────────
        results_tab2["universes"][universe_name] = {
            "full_ranking": [
                [t, z, get_action(z)] for t, z in ranked
            ]
        }

        logger.info(f"   ✅ {universe_name}: {len(ticker_scores)} ETFs ranked")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    logger.info("\n💾 Saving JSON results...")
    tab1_path = f"llm_alpha_{run_date}.json"
    tab2_path = f"llm_alpha_breakdown_{run_date}.json"

    with open(tab1_path, "w") as f:
        json.dump(results_tab1, f, indent=2, default=str)
    with open(tab2_path, "w") as f:
        json.dump(results_tab2, f, indent=2, default=str)

    logger.info(f"   Saved: {tab1_path}")
    logger.info(f"   Saved: {tab2_path}")

    if token:
        logger.info("\n📤 Uploading results to HuggingFace...")
        try:
            api = HfApi(token=token)
            for path in [tab1_path, tab2_path]:
                api.upload_file(
                    path_or_fileobj=path,
                    path_in_repo=path,
                    repo_id=config.RESULTS_REPO,
                    token=token,
                    repo_type="dataset"
                )
            logger.info("   ✅ Upload complete!")
        except Exception as e:
            logger.error(f"   Upload failed: {e}")

    return {"tab1": results_tab1, "tab2": results_tab2}


if __name__ == "__main__":
    run_trainer()
