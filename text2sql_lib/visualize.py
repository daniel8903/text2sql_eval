import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

def load_benchmark_results(jsonl_path: str) -> pd.DataFrame:
    """Load benchmark results from a JSONL file into a DataFrame."""
    with open(jsonl_path, "r") as f:
        data = [json.loads(line) for line in f]
    return pd.DataFrame(data)

def visualize_benchmark_results(
    jsonl_path: str,
    show: bool = True,
    save_dir: Optional[str] = None
):
    """
    Visualize token counts, latency, durations, accuracy, and correlations from benchmark results.
    
    Args:
        jsonl_path: Path to the benchmark_results.jsonl file.
        show: Whether to display the plots interactively.
        save_dir: If provided, saves plots to this directory.
    """
    df = load_benchmark_results(jsonl_path)
    numeric_cols = [
        'tokens_prompt', 'tokens_completion', 'tokens_total',
        'latency_sec', 'total_duration', 'tokens_per_sec'
    ]
    accuracy_cols = ['match_exact', 'match_ast', 'llm_equivalent']

    plot_params = [
        # (column, color, title, xlabel)
        ('tokens_prompt', 'blue', 'Prompt Token Count', 'Prompt Tokens'),
        ('tokens_completion', 'green', 'Completion Token Count', 'Completion Tokens'),
        ('tokens_total', 'red', 'Total Token Count', 'Total Tokens'),
        ('latency_sec', 'purple', 'Latency Distribution (seconds)', 'Latency (s)'),
        ('total_duration', 'orange', 'Total Duration Distribution (ms)', 'Total Duration (ms)'),
        ('tokens_per_sec', 'teal', 'Tokens per Second Distribution', 'Tokens per Second'),
    ]
    for col, color, title, xlabel in plot_params:
        plt.figure(figsize=(8, 5))
        data = df[col] if col != 'total_duration' else df[col] / 1e6
        sns.histplot(data, kde=True, color=color)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{col}_hist.png")
        if show:
            plt.show()
        plt.close()

    # Accuracy metrics bar plots
    for col in accuracy_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df, palette='pastel')
        plt.title(f"{col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.tight_layout()
        if save_dir:
            plt.savefig(f"{save_dir}/{col}_bar.png")
        if show:
            plt.show()
        plt.close()

    # Accuracy rates summary
    acc_summary = {col: df[col].mean() for col in accuracy_cols}
    print("Accuracy rates:")
    for k, v in acc_summary.items():
        print(f"  {k}: {v:.2%}")

    # Correlation heatmap for numeric metrics
    plt.figure(figsize=(7, 5))
    corr = df[numeric_cols].copy()
    corr['total_duration'] = corr['total_duration'] / 1e6  # convert to ms for heatmap
    sns.heatmap(corr.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap of Metrics")
    plt.tight_layout()
    if save_dir:
        plt.savefig(f"{save_dir}/correlation_heatmap.png")
    if show:
        plt.show()
    plt.close()

    # Boxplots: Numeric metrics grouped by accuracy
    for acc_col in accuracy_cols:
        for num_col in numeric_cols:
            plt.figure(figsize=(7, 4))
            data = df.copy()
            if num_col == 'total_duration':
                data[num_col] = data[num_col] / 1e6
            sns.boxplot(x=acc_col, y=num_col, data=data, palette='Set2')
            plt.title(f"{num_col} by {acc_col}")
            plt.xlabel(acc_col)
            plt.ylabel(num_col + (" (ms)" if num_col == "total_duration" else ""))
            plt.tight_layout()
            if save_dir:
                plt.savefig(f"{save_dir}/{num_col}_by_{acc_col}_box.png")
            if show:
                plt.show()
            plt.close()

def visualize_benchmark_results2(jsonl_path: str):
    # Load data
    df = load_benchmark_results(jsonl_path)

    # --- Accuracy Metrics ---
    accuracy_cols = ['match_exact', 'match_ast', 'llm_equivalent']
    for col in accuracy_cols:
        plt.figure(figsize=(6, 4))
        sns.countplot(x=col, data=df)
        plt.title(f"{col} Distribution")
        plt.show()
    print(df[accuracy_cols].mean())

    # --- Performance Metrics ---
    plt.figure(figsize=(8, 6))
    sns.histplot(df['latency_sec'], kde=True)
    plt.title("Latency Distribution")
    plt.show()

    plt.figure(figsize=(8, 6))
    sns.histplot(df['tokens_per_sec'], kde=True)
    plt.title("Tokens per Second Distribution")
    plt.show()

    # --- Complexity Analysis ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='sql_complexity', y='latency_sec', data=df)
    plt.title("Latency vs. SQL Complexity")
    plt.show()

    # Grouped bar plot for accuracy vs complexity
    complexity_accuracy = df.groupby('sql_complexity')[accuracy_cols].mean().reset_index()
    complexity_accuracy.set_index('sql_complexity').plot(kind='bar', figsize=(10, 6))
    plt.title("Accuracy vs. SQL Complexity")
    plt.ylabel("Accuracy Rate")
    plt.show()

    # --- Correlation Analysis ---
    numeric_cols = ['tokens_prompt', 'tokens_completion', 'tokens_total', 'latency_sec', 'tokens_per_sec'] + accuracy_cols
    corr_matrix = df[numeric_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.show()