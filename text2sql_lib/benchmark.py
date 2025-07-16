"""
benchmark.py
============

End-to-end-Benchmark:

  • SQL-Generierung        (model_cfg)
  • Äquivalenz-Beurteilung (judge_model_cfg)

External helper modules:
    sql_utils.py        – normalize_sql(), extract_sql(), ...
    gpu_monitor.py      – GPUMonitorThread
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import OrderedDict
from typing import Dict, Any, List, Optional

import pandas as pd
from tqdm import tqdm

from .sql_utils import (
    normalize_sql,
    extract_sql,
    ast_equal_ignore_alias,
    extract_json_block,
    remove_think,
)
from .gpu_monitor import GPUMonitorThread
from .providers import get_llm, ModelConfig

# ----------------------------------------------------------------------
#   Hilfsfunktionen
# ----------------------------------------------------------------------
_COMPLEXITY_CLASSES = [
    "basic SQL",
    "aggregation",
    "single join",
    "window functions",
    "multiple_joins",
    "subqueries",
    "set operations",
]


def _ns_to_s(value: int | None) -> float | None:
    """Nanosekunden → Sekunden (None bleibt None)."""
    return value / 1e9 if value is not None else None


def _stratified_sample(df: pd.DataFrame, total: int) -> pd.DataFrame:
    """Ziehe eine stratifizierte Stichprobe über sql_complexity."""
    quota = {c: total // len(_COMPLEXITY_CLASSES) for c in _COMPLEXITY_CLASSES}
    collected: list[pd.DataFrame] = []
    remaining = 0

    # 1. Durchgang – so viel wie möglich pro Klasse
    for cls in _COMPLEXITY_CLASSES:
        bucket = df[df["sql_complexity"] == cls]
        need = quota[cls]
        take = min(len(bucket), need)
        if take > 0:
            collected.append(bucket.sample(take, random_state=42))
        remaining += need - take  # Fehlbetrag

    # 2. Durchgang – Rest aus dem verbliebenen Pool
    if remaining > 0:
        still_available = df.drop(pd.concat(collected, ignore_index=True).index)
        extra = still_available.sample(min(remaining, len(still_available)),
                                       random_state=42)
        collected.append(extra)

    return (
        pd.concat(collected, ignore_index=True)
        .sample(frac=1, random_state=42)        # mischen
        .reset_index(drop=True)
    )


# ----------------------------------------------------------------------
# 1.  Low-level helper – uniform .chat(...)
# ----------------------------------------------------------------------
def _run_llm_chat(
    llm,
    model_name: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    stream: bool = False,
) -> Dict[str, Any]:
    return llm.chat(
        model_name=model_name,
        messages=messages,
        temperature=temperature,
        stream=stream,
    )


# ----------------------------------------------------------------------
# 2.  Generation utilities
# ----------------------------------------------------------------------
def generate_sql_from_prompt(
    llm,
    prompt: str,
    context: str,
    model_cfg: ModelConfig,
) -> Dict[str, Any]:
    """Generate SQL and collect timing/token stats (Sekunden!)."""
    # ---------- build messages ---------------------------------------
    system_prompt = """
    You are an expert database engineer whose task is to translate an
    English question into a single, executable SQL statement.

    Rules
    2. Use standard ANSI SQL.
    3. Refer ONLY to tables / columns present in the provided schema.
    """
    
    user_prompt = (
        "### Context (database schema)\n"
        f"{context}\n\n"
        "### Question\n"
        f"{prompt}\n\n"
        "### SQL"
    )

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt},
    ]

    # ---------- LLM call ---------------------------------------------
    t0 = time.time()
    rsp = _run_llm_chat(
        llm,
        model_name=model_cfg.name,
        messages=messages,
        temperature=model_cfg.temperature,
        stream=False,
    )
    latency = time.time() - t0

    raw_completion = rsp["message"]["content"]
    cleaned = remove_think(raw_completion)
    extracted_sql = extract_sql(cleaned)

    # ggf. Ollama-spezifische Telemetrie
    total_duration_sec = _ns_to_s(rsp.get("total_duration"))
    load_duration_sec = _ns_to_s(rsp.get("load_duration"))
    prompt_eval_count = rsp.get("prompt_eval_count", 0)
    prompt_eval_sec = _ns_to_s(rsp.get("prompt_eval_duration"))
    eval_count = rsp.get("eval_count", 0)
    eval_duration_sec = _ns_to_s(rsp.get("eval_duration"))

    total_tokens = (prompt_eval_count + eval_count) or None

    return {
        "generated_sql_raw": raw_completion,
        "generated_sql_extracted": extracted_sql,
        "latency_sec": latency,
        "total_duration_sec": total_duration_sec,
        "load_duration_sec": load_duration_sec,
        "tokens_prompt": prompt_eval_count,
        "prompt_eval_sec": prompt_eval_sec,
        "tokens_completion": eval_count,
        "completion_eval_sec": eval_duration_sec,
        "tokens_total": total_tokens,
        "tokens_per_sec": total_tokens / latency if total_tokens and latency else None,
    }


def ask_llm_equivalence_judge(
    judge_llm,
    ref_sql: str,
    gen_sql: str,
    context: str,
    judge_cfg: ModelConfig,
):
    """LLM beurteilt semantische Äquivalenz zweier SQL-Queries."""
    system_prompt = (
        "You are an expert SQL analyst. Compare two SQL queries and "
        "determine if they are semantically equivalent (produce the same result). "
        "Respond ONLY with a JSON object containing:\n"
        "  'equivalent' (true/false) and 'explanation' (string)."
    )
    user_prompt = (
        "### Context (database schema)\n"
        f"{context}\n\n"
        "### Reference SQL\n"
        f"{ref_sql}\n\n"
        "### Generated SQL\n"
        f"{gen_sql}\n\n"
        "### Task\n"
        "Are the two queries semantically equivalent? "
        "Return your decision in the requested JSON format."
    )

    rsp = _run_llm_chat(
        judge_llm,
        model_name=judge_cfg.name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=judge_cfg.temperature,
        stream=False,
    )

    raw_answer = rsp["message"]["content"]
    cleaned = remove_think(raw_answer)

    try:
        json_block = extract_json_block(cleaned)
        data = json.loads(json_block)
        equivalent = bool(data.get("equivalent", False))
        explanation = str(data.get("explanation", ""))
    except Exception:
        equivalent = False
        explanation = f"[Parsing failed] {raw_answer}"

    return equivalent, explanation


# ----------------------------------------------------------------------
# 3.  Main Benchmark
# ----------------------------------------------------------------------
def run_benchmark(
    dataset_path: str,
    log_path: str,
    model_cfg: ModelConfig,
    *,
    judge_model_cfg: Optional[ModelConfig] = None,
    total_examples: int = 200,
    gpu_monitor_interval: float = 1.0,
    monitor_gpu: bool = False,
):
    """
    Parameters
    ----------
    total_examples : Gesamtzahl der Beispiele (stratifiziert über Komplexität)
    """
    # -------- Daten einlesen & Stichprobe -----------------------------
    df = pd.read_parquet(dataset_path)
    df = _stratified_sample(df, total_examples)

    gen_llm = get_llm(model_cfg)
    judge_cfg = judge_model_cfg or model_cfg
    judge_llm = gen_llm if judge_model_cfg is None else get_llm(judge_cfg)

    if os.path.exists(log_path):
        os.remove(log_path)

    # -------- GPU-Monitoring -----------------------------------------
    gpu_monitor = None
    if monitor_gpu:
        gpu_monitor = GPUMonitorThread(interval_sec=gpu_monitor_interval)
        gpu_monitor.start()

    # -------- Iteration ----------------------------------------------
    loop = tqdm(list(df.iterrows()), total=len(df), desc="Benchmarking")

    for i, row in loop:
        prompt = row["sql_prompt"]
        context = row["sql_context"]
        reference_sql = row["sql"]
        example_id = int(row["id"])
        sql_complexity = row.get("sql_complexity")

        try:
            # ---- Generierung ----------------------------------------
            result = generate_sql_from_prompt(gen_llm, prompt, context, model_cfg)
            generated_sql = result["generated_sql_extracted"]

            # ---- ground-truth --------------------------------------
            match_exact = normalize_sql(generated_sql) == normalize_sql(reference_sql)
            match_ast = ast_equal_ignore_alias(generated_sql, reference_sql)

            # ---- LLM-Judge -----------------------------------------
            llm_equiv, llm_explanation = ask_llm_equivalence_judge(
                judge_llm, reference_sql, generated_sql, context, judge_cfg
            )

            # ---- Logging -------------------------------------------
            log_entry = {
                "example_id": example_id,
                "sql_complexity": sql_complexity,
                "prompt": prompt,
                "context": context,
                "reference_sql": reference_sql,
                "generated_sql": generated_sql,
                "raw_model_output": result["generated_sql_raw"],
                "latency_sec": result["latency_sec"],
                "total_duration_sec": result["total_duration_sec"],
                "load_duration_sec": result["load_duration_sec"],
                "tokens_prompt": result["tokens_prompt"],
                "prompt_eval_sec": result["prompt_eval_sec"],
                "tokens_completion": result["tokens_completion"],
                "completion_eval_sec": result["completion_eval_sec"],
                "tokens_total": result["tokens_total"],
                "tokens_per_sec": result["tokens_per_sec"],
                "match_exact": match_exact,
                "match_ast": match_ast,
                "llm_equivalent": llm_equiv,
                "llm_explanation": llm_explanation,
            }
            with open(log_path, "a") as fp:
                fp.write(json.dumps(log_entry) + "\n")

            loop.set_postfix(
                OrderedDict(
                    id=example_id,
                    exact=match_exact,
                    ast=match_ast,
                    llm=llm_equiv,
                )
            )

        except Exception as exc:
            print(f"[{i + 1}/{len(df)}] Error on id={example_id}: {exc}")

    # -------- GPU-Monitor schließen ----------------------------------
    if monitor_gpu and gpu_monitor:
        gpu_monitor.stop()
        gpu_monitor.join()
        gpu_log_path = log_path.replace(".jsonl", "_gpu_timeseries.json")
        with open(gpu_log_path, "w") as fp:
            json.dump(gpu_monitor.samples, fp, indent=2)
