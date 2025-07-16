from collections import OrderedDict
import os
import json
import re
import pandas as pd
from tqdm import tqdm
from ollama import Client
from .sql_utils import normalize_sql, extract_sql, ast_equal_ignore_alias, extract_json_block
from .gpu_monitor import GPUMonitorThread

def generate_sql_from_prompt(client, prompt, context, model_name):
    import time
    full_prompt = f"{context}\n\n--\n\n{prompt}"
    start_time = time.time()
    response = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
        options={"temperature": 0.1, "stream": False},
    )
    latency = time.time() - start_time

    raw_completion = response["message"]["content"]
    extracted_sql = extract_sql(raw_completion)

    total_duration = response.get("total_duration", None)
    load_duration = response.get("load_duration", None)
    prompt_eval_count = response.get("prompt_eval_count", 0)
    prompt_eval_duration = response.get("prompt_eval_duration", None)
    eval_count = response.get("eval_count", 0)
    eval_duration = response.get("eval_duration", None)

    total_tokens = prompt_eval_count + eval_count

    return {
        "generated_sql_raw": raw_completion,
        "generated_sql_extracted": extracted_sql,
        "latency_sec": latency,
        "total_duration": total_duration,
        "load_duration": load_duration,
        "tokens_prompt": prompt_eval_count,
        "prompt_eval_duration": prompt_eval_duration,
        "tokens_completion": eval_count,
        "eval_duration": eval_duration,
        "tokens_total": total_tokens,
        "tokens_per_sec": total_tokens / latency if latency > 0 else None,
    }

def ask_llm_equivalence_judge(client, ref_sql, gen_sql, context, model_name):
    system_prompt = (
        "You are an expert SQL analyst. Compare two SQL queries and "
        "determine if they are semantically equivalent (produce the same result). Given the following Context:\n"
        "### Context\n"
        f"{context}\n"
        "###\n"
        "Respond with a JSON object with keys: 'explanation' (text), 'equivalent' (true/false)."
    )
    user_prompt = (
        f"Reference SQL query:\n{ref_sql}\n\n"
        f"Generated SQL query:\n{gen_sql}\n\n"
        "Are these queries semantically equivalent? Answer in the specified JSON format."
    )

    full_prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = client.chat(
        # model=model_name,
        model="phi4:latest",
        options={"temperature": 0.3},
        messages=full_prompt,
        stream=False
    )

    raw_answer = response["message"]["content"]

    # Remove <think>...</think> blocks before trying to parse JSON
    cleaned_answer = re.sub(r"<think>.*?</think>", "", raw_answer, flags=re.DOTALL)

    try:
        clean_json = extract_json_block(cleaned_answer)
        result = json.loads(clean_json)
        equivalent = result.get("equivalent", False)
        explanation = result.get("explanation", "")
    except Exception:
        equivalent = False
        explanation = f"[Parsing failed] {raw_answer}"

    return equivalent, explanation


def run_benchmark(
    dataset_path,
    log_path,
    model_name,
    max_examples=5,
    gpu_monitor_interval=1.0,
    monitor_gpu=True
):
    df = pd.read_parquet(dataset_path)
    client = Client()

    if os.path.exists(log_path):
        os.remove(log_path)

    gpu_monitor = None
    if monitor_gpu:
        gpu_monitor = GPUMonitorThread(interval_sec=gpu_monitor_interval)
        gpu_monitor.start()

    from tqdm import tqdm
    loop = tqdm(list(df.head(max_examples).iterrows()), total=min(max_examples, len(df)), desc="Benchmarking")
    for i, row in loop:
        prompt = row["sql_prompt"]
        context = row["sql_context"]
        reference_sql = row["sql"]
        example_id = int(row["id"])
        sql_complexity = row.get("sql_complexity", None)

        try:
            result = generate_sql_from_prompt(client, prompt, context, model_name)
            generated_sql = result["generated_sql_extracted"]

            match_exact = normalize_sql(generated_sql) == normalize_sql(reference_sql)
            match_ast = ast_equal_ignore_alias(generated_sql, reference_sql)
            # llm_equiv, llm_explanation = ask_llm_equivalence_judge(client, reference_sql, generated_sql, context, model_name)
            llm_equiv, llm_explanation = False, "Not implemented"

            log_entry = {
                "example_id": example_id,
                "sql_complexity": sql_complexity,
                "prompt": prompt,
                "context": context,
                "reference_sql": reference_sql,
                "generated_sql": generated_sql,
                "raw_model_output": result["generated_sql_raw"],
                "latency_sec": result["latency_sec"],
                "total_duration": result["total_duration"],
                "load_duration": result["load_duration"],
                "tokens_prompt": result["tokens_prompt"],
                "prompt_eval_duration": result["prompt_eval_duration"],
                "tokens_completion": result["tokens_completion"],
                "eval_duration": result["eval_duration"],
                "tokens_total": result["tokens_total"],
                "tokens_per_sec": result["tokens_per_sec"],
                "match_exact": match_exact,
                "match_ast": match_ast,
                "llm_equivalent": llm_equiv,
                "llm_explanation": llm_explanation,
            }

            with open(log_path, "a") as log_file:
                log_file.write(json.dumps(log_entry) + "\n")

            loop.set_postfix(OrderedDict([
                ("example_id", example_id),
                ("exact", match_exact),
                ("ast", match_ast),
                ("llm", llm_equiv)
            ]))

        except Exception as e:
            print(f"[{i+1}/{max_examples}] Error on example_id={example_id}: {str(e)}")

    if monitor_gpu and gpu_monitor:
        gpu_monitor.stop()
        gpu_monitor.join()
        gpu_log_path = log_path.replace(".jsonl", "_gpu_timeseries.json")
        with open(gpu_log_path, "w") as gpu_file:
            json.dump(gpu_monitor.samples, gpu_file, indent=2)