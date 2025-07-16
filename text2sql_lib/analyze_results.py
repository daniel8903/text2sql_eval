import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = "benchmark_results.jsonl"  # Change if needed

def load_results(path):
    with open(path, "r") as f:
        records = [json.loads(line) for line in f]
    return pd.DataFrame(records)

def plot_accuracy_by_complexity(df):
    metrics = df.groupby("sql_complexity")[["match_exact", "match_ast", "llm_equivalent"]].mean().reset_index()
    melted = metrics.melt(id_vars="sql_complexity", var_name="metric", value_name="accuracy")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="sql_complexity", y="accuracy", hue="metric")
    plt.title("Accuracy by SQL Complexity")
    plt.ylabel("Accuracy")
    plt.xlabel("SQL Complexity")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_llm_judging_error(df):
    df["gt_semantic"] = df["match_exact"] | df["match_ast"]
    df["llm_judging_error"] = df["llm_equivalent"] != df["gt_semantic"]
    err_rates = df.groupby("sql_complexity")["llm_judging_error"].mean().reset_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(data=err_rates, x="sql_complexity", y="llm_judging_error", color="salmon")
    plt.title("LLM Judging Error Rate by SQL Complexity")
    plt.ylabel("Error Rate")
    plt.xlabel("SQL Complexity")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_latency_by_complexity(df):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="sql_complexity", y="latency_sec", palette="Set3")
    plt.title("Generation Latency by SQL Complexity")
    plt.ylabel("Latency (seconds)")
    plt.xlabel("SQL Complexity")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(df):
    df["gt_semantic"] = df["match_exact"] | df["match_ast"]
    matrix = df.groupby(["sql_complexity", "llm_equivalent", "gt_semantic"]).size().reset_index(name="count")
    g = sns.catplot(
        data=matrix,
        kind="bar",
        x="sql_complexity",
        y="count",
        hue="llm_equivalent",
        col="gt_semantic",
        height=5,
        aspect=1.2
    )
    g.set_titles("Ground Truth: {col_name}")
    g.set_axis_labels("SQL Complexity", "Example Count")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("LLM Judgment vs Ground Truth by SQL Complexity")
    plt.show()

def main():
    df = load_results(RESULTS_PATH)
    print("Columns:", df.columns)
    plot_accuracy_by_complexity(df)
    plot_llm_judging_error(df)
    plot_latency_by_complexity(df)
    plot_confusion_matrix(df)

if __name__ == "__main__":
    sns.set(style="whitegrid", palette="muted", font_scale=1.1)
    main()