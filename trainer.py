"""
trainer.py  —  Orchestrator for LLM Alpha Suite
================================================

Loads data → runs multi-agent system → generates signals → builds JSON.
Uses parallel processing for speed.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

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


def process_window(args: Tuple) -> Dict:
    """Process a single window for a universe in parallel."""
    window, universe_name, available, prices_df, macro_df, config_dict = args
    
    try:
        universe_prices = prices_df[available]
        
        # ── Results for this window ──────────────────────────────────────────
        ticker_scores = {}
        ticker_details = {}
        ticker_agent_signals = {}
        ticker_alpha_signals = {}
        ticker_rl_signals = {}
        ticker_combined = {}
        
        for ticker in available:
            prices = universe_prices[ticker].values
            returns = np.log(prices / np.roll(prices, 1))[1:]
            
            if len(returns) < window:
                ticker_scores[ticker] = 0
                continue
            
            data = {
                "prices": prices[-window:],
                "returns": returns[-window:],
                "ticker": ticker
            }
            
            # ── Module 1: Multi-Agent Alpha Discovery ──────────────────────
            agent_orchestrator = AgentOrchestrator(config_dict)
            agent_result = agent_orchestrator.analyze(data)
            agent_signal = agent_result.get("signal", 0)
            
            # ── Module 2: Symbolic Alpha ──────────────────────────────────────
            alpha_generator = SymbolicAlphaGenerator(config_dict.get("symbolic_alpha", {}))
            alphas = alpha_generator.generate_alphas(data)
            evaluated = alpha_generator.evaluate_alphas(alphas, returns[-window:]) if alphas else []
            alpha_signal = evaluated[0]["signal"][-1] if evaluated else 0
            
            # ── Module 3: LLM-Guided RL ──────────────────────────────────────
            llm_rl_agent = LLMRLAgent(config_dict.get("llm_rl", {}))
            rl_result = llm_rl_agent.select_action(data)
            rl_signal = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}.get(rl_result["action"], 0.0)
            
            # ── Module 4: Signal Combination ──────────────────────────────────
            signals = {
                "multi_agent": {"signal": agent_signal, "confidence": agent_result.get("confidence", 0.5)},
                "symbolic_alpha": {"signal": alpha_signal, "confidence": 0.5},
                "llm_rl": {"signal": rl_signal, "confidence": rl_result.get("llm_proposal", {}).get("confidence", 0.5)}
            }
            
            signal_combiner = SignalCombiner(config_dict.get("evaluator", {}))
            combined = signal_combiner.combine(signals)
            final_signal = combined["signal"]
            
            # Store results
            z_score = final_signal * 5
            ticker_scores[ticker] = z_score
            ticker_details[ticker] = {
                "agent_signal": agent_signal,
                "alpha_signal": alpha_signal,
                "rl_signal": rl_signal,
                "final_signal": final_signal
            }
            ticker_agent_signals[ticker] = agent_signal
            ticker_alpha_signals[ticker] = alpha_signal
            ticker_rl_signals[ticker] = rl_signal
            ticker_combined[ticker] = final_signal
        
        # ── Normalize z-scores ──────────────────────────────────────────────
        z_values = np.array(list(ticker_scores.values()))
        if len(z_values) > 1 and np.std(z_values) > 1e-6:
            mean_z = np.mean(z_values)
            std_z = np.std(z_values)
            for ticker in ticker_scores:
                ticker_scores[ticker] = (ticker_scores[ticker] - mean_z) / std_z
        
        return {
            "window": window,
            "universe": universe_name,
            "ticker_scores": ticker_scores,
            "ticker_details": ticker_details,
            "error": None
        }
    except Exception as e:
        return {
            "window": window,
            "universe": universe_name,
            "ticker_scores": {},
            "ticker_details": {},
            "error": str(e)
        }


def run_trainer(hf_token: Optional[str] = None) -> Dict:
    """Run the full LLM Alpha Suite pipeline."""
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

    engine_config = {
        "agents": config.AGENTS,
        "symbolic_alpha": config.SYMBOLIC_ALPHA,
        "llm_rl": config.LLM_RL,
        "evaluator": config.EVALUATOR,
    }

    results_tab1 = {"run_date": run_date, "universes": {}}
    results_tab2 = {"run_date": run_date, "universes": {}}

    # ── Prepare parallel tasks ───────────────────────────────────────────────
    tasks = []
    windows = config.WINDOWS
    max_workers = max(1, int(mp.cpu_count() * 0.75))
    logger.info(f"🚀 Using {max_workers} parallel workers")

    for universe_name, tickers in config.UNIVERSES.items():
        available = [t for t in tickers if t in prices_df.columns]
        if not available:
            continue

        for window in windows:
            tasks.append((window, universe_name, available, prices_df, macro_df, engine_config))

    logger.info(f"📋 Total tasks: {len(tasks)}")
    all_results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {executor.submit(process_window, task): task for task in tasks}
        completed = 0
        for future in as_completed(future_to_task):
            completed += 1
            try:
                result = future.result(timeout=1800)
                if result.get("error"):
                    logger.warning(f"   ⚠️ {result['universe']} @ {result['window']}d failed: {result['error']}")
                    continue
                key = f"{result['universe']}_{result['window']}"
                all_results[key] = result
                logger.info(f"   ✅ [{completed}/{len(tasks)}] {result['universe']} @ {result['window']}d")
            except Exception as e:
                logger.error(f"   ❌ Task failed: {e}")

    logger.info(f"✅ Completed {len(all_results)}/{len(tasks)} tasks")

    # ── Build results ──────────────────────────────────────────────────────────
    for universe_name in config.UNIVERSES.keys():
        available = [t for t in config.UNIVERSES[universe_name] if t in prices_df.columns]
        if not available:
            continue

        # Collect results for this universe across all windows
        universe_results = {}
        for key, result in all_results.items():
            if result.get("universe") == universe_name:
                universe_results[str(result["window"])] = result

        if not universe_results:
            continue

        # ── Tab 1: Best window per ETF ──────────────────────────────────────
        best_window_per_etf = {}
        for ticker in available:
            best_z = -999
            best_win = None
            best_data = None
            for window, wr in universe_results.items():
                ticker_data = wr.get("ticker_details", {}).get(ticker, {})
                z = safe_float(wr.get("ticker_scores", {}).get(ticker, -999))
                if z > best_z:
                    best_z = z
                    best_win = window
                    best_data = ticker_data
            if best_win is not None:
                best_window_per_etf[ticker] = {
                    "z_score": best_z,
                    "window": int(best_win),
                    "agent_signal": safe_float(best_data.get("agent_signal", 0)),
                    "alpha_signal": safe_float(best_data.get("alpha_signal", 0)),
                    "rl_signal": safe_float(best_data.get("rl_signal", 0)),
                    "final_signal": safe_float(best_data.get("final_signal", 0)),
                    "action": get_action(best_z)
                }

        if not best_window_per_etf:
            continue

        top_buys = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1], reverse=True
        )[:5]

        top_sells = sorted(
            [(t, d["z_score"]) for t, d in best_window_per_etf.items()],
            key=lambda x: x[1]
        )[:5]

        results_tab1["universes"][universe_name] = {
            "top_buys": [{"ticker": t, "z_score": z} for t, z in top_buys],
            "top_sells": [{"ticker": t, "z_score": z} for t, z in top_sells],
            "full_scores": best_window_per_etf
        }

        # ── Tab 2: Per-window breakdown ──────────────────────────────────────
        windows_data = {}
        for window, wr in universe_results.items():
            ticker_scores = wr.get("ticker_scores", {})
            
            # Build top buys for this window
            top_buys_window = sorted(
                [(t, z) for t, z in ticker_scores.items()],
                key=lambda x: x[1], reverse=True
            )[:5]
            
            # Build full ranking for this window
            full_ranking = [
                [
                    t,
                    safe_float(z),
                    get_action(safe_float(z))
                ]
                for t, z in ticker_scores.items()
            ]
            
            windows_data[window] = {
                "top_buys": [{"ticker": t, "z_score": z} for t, z in top_buys_window],
                "full_ranking": full_ranking
            }
        
        results_tab2["universes"][universe_name] = {
            "windows": windows_data
        }

        logger.info(f"   ✅ {universe_name}: {len(best_window_per_etf)} ETFs ranked across {len(windows_data)} windows")

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
