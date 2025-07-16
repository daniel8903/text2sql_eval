"""
CLI entry-point for the text-to-SQL benchmark.

Usage (default everything):
    python -m text2sql_lib.cli qwen2.5:latest

Same but judge with GPT-4o via Azure:
    python -m text2sql_lib.cli qwen2.5:latest --judge gpt-4o --judge-provider azure

Full control (example):
    python -m text2sql_lib.cli qwen2.5:latest \
        --provider ollama --host http://192.168.178.187:11434 \
        --dataset data/my.parquet --log logs/myrun.jsonl \
        --judge qwen3-30b-a3b --judge-provider siemens --judge-temp 0.15
"""
from __future__ import annotations

import datetime as dt
import pathlib
import os
import typer

from text2sql_lib.providers import ModelProvider, ModelConfig
from text2sql_lib.benchmark import run_benchmark

app = typer.Typer(add_completion=False)

# ------------ global defaults -------------------------------------------------
DEFAULT_DATASET = pathlib.Path("synthetic_text_to_sql/cached_test.parquet")
DEFAULT_JUDGE   = "qwen3-30b-a3b"
DEFAULT_JUDGE_PROVIDER = "siemens"
DEFAULT_LOG_DIR = pathlib.Path("benchmark_logs")
DEFAULT_GEN_PROVIDER = "ollama"
DEFAULT_GEN_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

PROVIDER_MAP = {
    "azure":   ModelProvider.AZURE,
    "siemens": ModelProvider.SIEMENS,
    "ollama":  ModelProvider.OLLAMA,
}


def _provider(value: str) -> ModelProvider:
    try:
        return PROVIDER_MAP[value.lower()]
    except KeyError:
        raise typer.BadParameter("provider must be azure | siemens | ollama") from None


@app.command(context_settings={"allow_extra_args": False})
def main(
    # ------ positional: the only argument you often change -------------
    gen_name: str = typer.Argument(
        ...,
        help="Generator model name (e.g. 'qwen2.5:latest', 'gpt-4o').",
    ),

    # ------ occasionally changed ---------------------------------------
    provider: str = typer.Option(
        DEFAULT_GEN_PROVIDER,
        "--provider", "-p",
        help=f"(generator) provider [{', '.join(PROVIDER_MAP)}]; "
             f"default: {DEFAULT_GEN_PROVIDER}",
    ),
    host: str | None = typer.Option(
        DEFAULT_GEN_HOST,
        "--host",
        help=f"(generator) Ollama URL; default: {DEFAULT_GEN_HOST}",
    ),
    temp: float = typer.Option(
        0.1, "--temp", help="(generator) temperature", show_default=True
    ),

    # ------ judge -------------------------------------------------------
    judge: str | None = typer.Option(
        DEFAULT_JUDGE,
        "--judge",
        help=f"Judge model name; default: {DEFAULT_JUDGE} "
             f'("{DEFAULT_JUDGE_PROVIDER}" provider). '
             "Use '' to reuse generator as judge.",
    ),
    judge_provider: str = typer.Option(
        DEFAULT_JUDGE_PROVIDER,
        "--judge-provider",
        help=f"Provider for judge model; default: {DEFAULT_JUDGE_PROVIDER}",
    ),
    judge_temp: float = typer.Option(0.2, "--judge-temp", help="Judge temperature"),
    judge_host: str | None = typer.Option(
        None, "--judge-host", help="Judge Ollama URL (if provider=ollama)"
    ),

    # ------ rarely changed ---------------------------------------------
    dataset: pathlib.Path = typer.Option(
        DEFAULT_DATASET, "--dataset", help="Parquet file with prompts"
    ),
    log: pathlib.Path | None = typer.Option(
        None,
        "--log",
        help=(
            "JSONL log file. If omitted, a name is auto-generated in "
            f"{DEFAULT_LOG_DIR}"
        ),
    ),
    max_examples: int = typer.Option(
        5, "--max-examples", "-n", help="How many rows to run from dataset"
    ),
):
    """Run the benchmark with a one-liner."""
    # ----------- derive configs ----------------------------------------
    gen_cfg = ModelConfig(
        name=gen_name,
        provider=_provider(provider),
        temperature=temp,
        host=host,
    )

    judge_cfg = None
    if judge:
        judge_cfg = ModelConfig(
            name=judge,
            provider=_provider(judge_provider),
            temperature=judge_temp,
            host=judge_host,
        )

    # ----------- auto-generate log file name if not provided -----------
    if log is None:
        DEFAULT_LOG_DIR.mkdir(exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        safe_gen = gen_name.replace("/", "-").replace(":", "-")
        log = DEFAULT_LOG_DIR / f"{safe_gen}_{ts}.jsonl"

    # ----------- run benchmark ----------------------------------------
    run_benchmark(
        dataset_path=str(dataset),
        log_path=str(log),
        model_cfg=gen_cfg,
        judge_model_cfg=judge_cfg,
        total_examples=max_examples,
    )

    typer.echo(f"\n✅  Finished. Log written to {log}")


if __name__ == "__main__":
    app()
