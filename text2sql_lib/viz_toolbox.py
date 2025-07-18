# viz_toolbox.py
from collections import defaultdict
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from typing import Dict, List, Optional

from .result_metrics import LLMResultMetrics

# # ──────────────────────────────────────────────────────────────────────────
# # Helper – turn the metrics-dict (var_name → LLMResultMetrics) into nice DF
# # ──────────────────────────────────────────────────────────────────────────
# _LEETCODE_CLASSES = [
#     "basic SQL", "aggregation", "single join",
#     "window functions", "multiple_joins", "subqueries", "set operations"
# ]
# _COMPLEXITY_CLASSES = ["easy", "medium", "hard"]


# def _build_df(metrics: Dict[str, "LLMResultMetrics"]) -> pd.DataFrame:
#     rows = []
#     for name, m in metrics.items():
#         rows.append(
#             dict(
#                 model=name,
#                 total=m.total_queries,
#                 exact=m.accuracy_exact(),
#                 ast=m.accuracy_ast(),
#                 tech=m.accuracy_judge_technical(),
#                 intent=m.accuracy_judge_intent(),
#                 superior_cnt=len(m.superior_ids),
#                 cost=m.cost_eur,
#                 latency=m.avg_latency_sec,
#                 **{f"diff_{lc}": m.accuracy_exact(lc) for lc in _LEETCODE_CLASSES},
#                 **{c: m.accuracy_exact(c) for c in _COMPLEXITY_CLASSES},
#             )
#         )
#     return pd.DataFrame(rows).set_index("model")


# # ──────────────────────────────────────────────────────────────────────────
# # 1. Overall accuracy bar-chart
# # ──────────────────────────────────────────────────────────────────────────
# def plot_overall(df: pd.DataFrame, metric: str = "intent") -> None:
#     """
#     metric ∈ {exact, ast, tech, intent}
#     """
#     ax = (df[metric] * 100).sort_values(ascending=False).plot.barh(
#         figsize=(6, 0.4 * len(df) + 1),
#         color="steelblue",
#     )
#     ax.set_xlabel(f"{metric} accuracy (%)")
#     ax.set_ylabel("model")
#     ax.set_xlim(0, 100)
#     for p in ax.patches:
#         ax.annotate(f"{p.get_width():.1f}%", (p.get_width() + 1, p.get_y() + 0.1))
#     plt.gca().invert_yaxis()
#     plt.tight_layout()

# _CMAP = {
#     "exact":      "#1f77b4",   # blue
#     "ast":        "#ff7f0e",   # orange
#     "tech":       "#2ca02c",   # green
#     "intent":     "#d62728",   # red
# }

# def plot_overall_multi(df: pd.DataFrame) -> None:
#     """
#     Four horizontally aligned bar-plots (Exact / AST / Tech / Intent).
#     Model order is defined by descending *Intent* accuracy.
#     """
#     metrics = ["intent", "tech", "ast", "exact"]

#     # ── ordering of rows by intent accuracy ────────────────────────────
#     order = df["intent"].sort_values(ascending=False).index
#     df_sorted = df.loc[order]

#     # ── build the 4-subplot canvas ─────────────────────────────────────
#     n_models = len(df_sorted)
#     fig, axes = plt.subplots(
#         nrows=1, ncols=4,
#         figsize=(14, 0.35 * n_models + 1),
#         sharey=True
#     )

#     for ax, metric in zip(axes, metrics):
#         (df_sorted[metric] * 100).plot.barh(
#             ax=ax,
#             color=_CMAP[metric],
#         )
#         ax.set_title(metric.capitalize())
#         ax.set_xlim(0, 100)
#         ax.set_xlabel("%")
#         ax.set_ylabel("")           # shared y-axis; no duplicate label
#         ax.invert_yaxis()           # highest value on top
#         for p in ax.patches:
#             ax.annotate(
#                 f"{p.get_width():.1f}%",
#                 (p.get_width() + 1, p.get_y() + 0.15),
#                 fontsize=8,
#             )

#     axes[0].set_ylabel("model")     # only leftmost subplot shows label
#     plt.tight_layout()

# # ---------------------------------------------------------------------------
# # metrics_dict  :  {"deepseek_r1_14b": LLMResultMetrics, ...}
# # metric_fn_name:  "accuracy_exact" | "accuracy_ast" | "accuracy_judge_technical"
# # ---------------------------------------------------------------------------
# def plot_difficulty_grouped(
#     metrics_dict: Dict[str, "LLMResultMetrics"],
#     metric_fn_name: str = "accuracy_judge_technical",
# ) -> None:
#     """
#     Drei horizontale Balken (easy / medium / hard) pro Modell,
#     sortiert absteigend nach der *globalen* Accuracy des gewünschten Metrics.
#     """

#     # 1) Daten sammeln ----------------------------------------------------
#     rows = []
#     for name, m in metrics_dict.items():
#         fn = getattr(m, metric_fn_name)          # gewünschte Accuracy-Methode
#         rows.append(
#             dict(
#                 model=name,
#                 total=fn(),                      # globale Accuracy
#                 easy=fn("easy"),
#                 medium=fn("medium"),
#                 hard=fn("hard"),
#             )
#         )

#     # in DataFrame oder Liste überführen
#     rows.sort(key=lambda r: r["total"], reverse=True)   # sortieren nach total

#     # 2) Plot vorbereiten --------------------------------------------------
#     y_pos  = np.arange(len(rows))          # 0,1,2,…
#     bar_h  = 0.25                          # Höhe je Balken

#     plt.figure(figsize=(10, 0.6 * len(rows) + 1))

#     # 3 Balken je Modell
#     plt.barh(y_pos + bar_h, [r["easy"]*100  for r in rows],
#              height=bar_h, color="#00c853", label="Easy")
#     plt.barh(y_pos,          [r["medium"]*100 for r in rows],
#              height=bar_h, color="#ffca28", label="Medium")
#     plt.barh(y_pos - bar_h, [r["hard"]*100  for r in rows],
#              height=bar_h, color="#d50000", label="Hard")

#     # Achsen & Beschriftungen
#     plt.yticks(y_pos, [r["model"] for r in rows])
#     plt.xlim(0, 100)
#     plt.xlabel(f"{metric_fn_name} (%)")
#     plt.title(f"{metric_fn_name} nach Schwierigkeitsgrad")
#     plt.legend(loc="lower right")

#     # Balken beschriften
#     for offset, key in [(bar_h, "easy"), (0, "medium"), (-bar_h, "hard")]:
#         for y, r in zip(y_pos, rows):
#             val = r[key] * 100
#             plt.annotate(f"{val:.1f}%",
#                          (val + 1, y + offset + bar_h / 4),
#                          fontsize=8)

#     plt.gca().invert_yaxis()
#     plt.tight_layout()

# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Dict, List

# # Farbpalette
# _EASY_COL, _MED_COL, _HARD_COL = "#00c853", "#ffca28", "#d50000"
# _METRIC_ORDER: List[str] = [
#     "accuracy_judge_intent",
#     "accuracy_judge_technical",
#     "accuracy_ast",
#     "accuracy_exact",
# ]
# _METRIC_TITLES = {
#     "accuracy_judge_intent":      "Intent",
#     "accuracy_judge_technical":   "Tech",
#     "accuracy_ast":               "AST",
#     "accuracy_exact":             "Exact",
# }


# def plot_difficulty_multi_metrics(metrics_dict: Dict[str, "LLMResultMetrics"]) -> None:
#     """
#     Vier Subplots (Intent / Tech / AST / Exact).
#     Jeder Subplot: drei Balken (Easy / Medium / Hard) pro Modell.
#     Modelle sind nach Intent-Accuracy (gesamt) absteigend sortiert.
#     """

#     # ------------------------------------------------------------------ #
#     # 1) Daten einsammeln
#     # ------------------------------------------------------------------ #
#     rows = {}
#     for name, m in metrics_dict.items():
#         rows[name] = {
#             "intent":     m.accuracy_judge_intent(),
#             "tech":       m.accuracy_judge_technical(),
#             "ast":        m.accuracy_ast(),
#             "exact":      m.accuracy_exact(),
#             "easy": {
#                 fn: getattr(m, fn)("easy") for fn in _METRIC_ORDER
#             },
#             "medium": {
#                 fn: getattr(m, fn)("medium") for fn in _METRIC_ORDER
#             },
#             "hard": {
#                 fn: getattr(m, fn)("hard") for fn in _METRIC_ORDER
#             },
#         }

#     # Sortieren nach Intent-Accuracy
#     model_order = sorted(rows.keys(), key=lambda k: rows[k]["intent"], reverse=True)
#     n_models = len(model_order)
#     y_pos = np.arange(n_models)
#     bar_h = 0.22

#     # ------------------------------------------------------------------ #
#     # 2) Plot-Canvas
#     # ------------------------------------------------------------------ #
#     fig, axes = plt.subplots(
#         nrows=1, ncols=4,
#         figsize=(15, 0.45 * n_models + 1),
#         sharey=True
#     )

#     for ax, metric_fn in zip(axes, _METRIC_ORDER):
#         # Easy / Medium / Hard Werte je Modell
#         easy   = [rows[m]["easy"][metric_fn]   * 100 for m in model_order]
#         medium = [rows[m]["medium"][metric_fn] * 100 for m in model_order]
#         hard   = [rows[m]["hard"][metric_fn]   * 100 for m in model_order]

#         # Drei Balken je Modell
#         ax.barh(y_pos + bar_h, easy,   bar_h, color=_EASY_COL, label="Easy")
#         ax.barh(y_pos,          medium, bar_h, color=_MED_COL,  label="Medium")
#         ax.barh(y_pos - bar_h, hard,   bar_h, color=_HARD_COL,  label="Hard")

#         ax.set_title(_METRIC_TITLES[metric_fn])
#         ax.set_xlim(0, 100)
#         ax.set_xlabel("%")
#         ax.invert_yaxis()

#         # Balkenbeschriftungen
#         for idx, (e, m_, h) in enumerate(zip(easy, medium, hard)):
#             for val, off in [(e, bar_h), (m_, 0), (h, -bar_h)]:
#                 if val > 0:
#                     ax.annotate(f"{val:.1f}%",
#                                 (val + 1, idx + off + bar_h / 4),
#                                 fontsize=7)

#         # y-Ticks nur links
#         if ax is axes[0]:
#             ax.set_yticks(y_pos)
#             ax.set_yticklabels(model_order)
#             ax.legend(loc="lower right", fontsize=8)
#         else:
#             ax.set_yticks([])

#     plt.tight_layout()
#     plt.show()

# # ──────────────────────────────────────────────────────────────────────────
# # 2. Easy / Medium / Hard stacked bars
# # ──────────────────────────────────────────────────────────────────────────
# def plot_difficulty_split(df: pd.DataFrame) -> None:
#     diff_df = df[[f"diff_{c}" for c in _COMPLEXITY_CLASSES]] * 100
#     diff_df.columns = _COMPLEXITY_CLASSES
#     diff_df.sort_index(inplace=True)

#     diff_df.plot(kind="barh",
#                  stacked=True,
#                  color=["#00c853", "#ffca28", "#d50000"],
#                  figsize=(6, 0.4 * len(diff_df) + 1))
#     plt.legend(loc="lower right", bbox_to_anchor=(1.25, 0))
#     plt.xlabel("Exact accuracy per difficulty (%)")
#     plt.xlim(0, 100)
#     plt.tight_layout()


# # ──────────────────────────────────────────────────────────────────────────
# # 3. SQL-class heat-map
# # ──────────────────────────────────────────────────────────────────────────
# def plot_sql_heatmap(df: pd.DataFrame) -> None:
#     sql_df = df[_LEETCODE_CLASSES] * 100
#     plt.figure(figsize=(10, 0.5 * len(df) + 1))
#     sns.heatmap(sql_df,
#                 annot=True,
#                 fmt=".1f",
#                 cmap="Blues",
#                 linewidths=0.5,
#                 cbar_kws={"label": "Exact accuracy (%)"})
#     plt.xlabel("")
#     plt.ylabel("model")
#     plt.tight_layout()

# # -----------------------------------------------------------------------
# #   visual: plot number of *superior* decisions per model
# # -----------------------------------------------------------------------
# def plot_superior(df: pd.DataFrame, *, show_counts: bool = True) -> None:
#     """
#     Horizontal bar-chart showing *percentage* of superior judgments.
#     Data-frame must contain columns  `total`  and  `superior_cnt`.
#     """
#     pct = (df["superior_cnt"] / df["total"] * 100).sort_values(ascending=False)

#     ax = pct.plot.barh(
#         figsize=(6, 0.4 * len(pct) + 1),
#         color="#8e24aa",
#     )
#     ax.set_xlabel("Superior judgments (%)")
#     ax.set_ylabel("model")
#     ax.set_xlim(0, 100)
#     ax.invert_yaxis()               # best model on top

#     # annotate each bar --------------------------------------------------
#     for bar, (model, pct_val) in zip(ax.patches, pct.items()):
#         raw = int(df.loc[model, "superior_cnt"])
#         txt = f"{pct_val:.1f}%"
#         if show_counts:
#             txt += f" ({raw})"
#         ax.annotate(
#             txt,
#             (bar.get_width() + 1, bar.get_y() + bar.get_height() / 4),
#             fontsize=9,
#         )

#     plt.tight_layout()

# # ──────────────────────────────────────────────────────────────────────────
# # 4. “Bang-for-buck” scatter: cost vs. overall intent accuracy
# # ──────────────────────────────────────────────────────────────────────────
# def plot_cost_vs_accuracy(df: pd.DataFrame) -> None:
#     ax = sns.scatterplot(
#         data=df,
#         x="cost",
#         y=df["intent"] * 100,
#         s=120,
#     )
#     for txt, row in df.iterrows():
#         ax.annotate(txt, (row["cost"], row["intent"] * 100),
#                     textcoords="offset points", xytext=(0, 5), ha="center")
#     ax.set_xlabel("Total evaluation cost (€)")
#     ax.set_ylabel("Intent accuracy (%)")
#     ax.set_title("Bang-for-buck")
#     plt.tight_layout()


# # ──────────────────────────────────────────────────────────────────────────
# # 5. Radar plot for a single model
# # ──────────────────────────────────────────────────────────────────────────
# def plot_radar(model_name: str, metrics_df: pd.DataFrame) -> None:
#     row = metrics_df.loc[model_name, _LEETCODE_CLASSES] * 100
#     # ─ matplotlib radar boiler-plate ─
#     labels = row.index.to_list()
#     values = row.to_list()
#     values += values[:1]                      # close polygon
#     angles = [n / float(len(labels)) * 2 * 3.14159 for n in range(len(labels))]
#     angles += angles[:1]

#     fig, ax = plt.subplots(subplot_kw=dict(polar=True))
#     ax.plot(angles, values, marker="o", color="teal")
#     ax.fill(angles, values, alpha=0.25, color="teal")
#     ax.set_thetagrids([a * 180/3.14159 for a in angles[:-1]], labels)
#     ax.set_ylim(0, 100)
#     ax.set_title(model_name)
#     plt.tight_layout()


# # ──────────────────────────────────────────────────────────────────────────
# # Example usage  (put this at bottom of your notebook / script)
# # ──────────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     from pathlib import Path
#     from complexity_map import complexity_map

#     BENCH_DIR = Path("better_benchmarks")

#     metrics_objects = {}
#     for f in BENCH_DIR.glob("*.jsonl"):
#         metrics_objects[f.stem] = LLMResultMetrics.from_jsonl(
#             f,
#             model_name=f.stem,
#             complexity_map=complexity_map,
#         )

#     df = _build_df(metrics_objects)

#     # Pick and choose the visual you want:
#     # plot_overall_multi(df)
#     # plt.show()

#     # plot_superior(df)
#     # plt.show()

#     # plot_difficulty_grouped(metrics, metric_fn_name="accuracy_judge_intent")
#     # plt.show()

#     # plot_difficulty_multi_metrics(metrics)
#     # plt.show()

#     # plot_difficulty_split(df)
#     # plt.show()

#     # plot_sql_heatmap(df)
#     # plt.show()

#     # plot_cost_vs_accuracy(df)
#     # plt.show()

#     # plot_radar("gpt-4.1", df)      # put any model-name you have
#     # plt.show()

def plot_overall_multi(metrics: Dict[str, LLMResultMetrics]) -> None:
    """
    Four horizontally aligned bar-plots (Exact / AST / Tech / Intent).
    Model order is defined by descending *Intent* accuracy.
    """
    # Create data dictionary for plotting
    data = {
        m.model_name: {
            "exact": m.accuracy_exact() * 100,
            "ast": m.accuracy_ast() * 100,
            "technical": m.accuracy_judge_technical() * 100,
            "intent": m.accuracy_judge_intent() * 100
        }
        for _, m in metrics.items()
    }

    # Convert to DataFrame and sort by intent accuracy
    df = pd.DataFrame.from_dict(data, orient='index')
    df_sorted = df.sort_values(by='intent', ascending=False)

    # Define metrics and their order
    metric_cols = ["intent", "technical", "ast", "exact"]

    # Create the plot
    n_models = len(df_sorted)
    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(14, 0.35 * n_models + 1),
        sharey=True
    )

    # Color map for different metrics
    _CMAP = {
        "intent": "#2ecc71",
        "technical": "#3498db",
        "ast": "#e74c3c",
        "exact": "#f1c40f"
    }

    # Plot each metric
    for ax, metric in zip(axes, metric_cols):
        df_sorted[metric].plot.barh(
            ax=ax,
            color=_CMAP[metric],
        )
        ax.set_title(metric.capitalize())
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy %")
        ax.set_ylabel("")
        ax.invert_yaxis()
        
        # Add value labels
        for p in ax.patches:
            ax.annotate(
                f"{p.get_width():.1f}%",
                (p.get_width() + 1, p.get_y() + 0.15),
                fontsize=8,
            )

    # Set ylabel only for leftmost subplot
    axes[0].set_ylabel("model")
    plt.tight_layout()

def plot_difficulty_grouped(
    metrics_dict: Dict[str, "LLMResultMetrics"],
    metric_fn_name: str = "accuracy_judge_technical",
) -> None:
    """
    Drei horizontale Balken (easy / medium / hard) pro Modell,
    sortiert absteigend nach der *globalen* Accuracy des gewünschten Metrics.
    """

    # 1) Daten sammeln ----------------------------------------------------
    rows = []
    for name, m in metrics_dict.items():
        fn = getattr(m, metric_fn_name)          # gewünschte Accuracy-Methode
        rows.append(
            dict(
                model=m.model_name,
                total=fn(),                      # globale Accuracy
                easy=fn("easy"),
                medium=fn("medium"),
                hard=fn("hard"),
            )
        )

    # in DataFrame oder Liste überführen
    rows.sort(key=lambda r: r["total"], reverse=True)   # sortieren nach total

    # 2) Plot vorbereiten --------------------------------------------------
    y_pos  = np.arange(len(rows))          # 0,1,2,…
    bar_h  = 0.25                          # Höhe je Balken

    plt.figure(figsize=(10, 0.6 * len(rows) + 1))

    # 3 Balken je Modell
    plt.barh(y_pos + bar_h, [r["easy"]*100  for r in rows],
             height=bar_h, color="#00c853", label="Easy")
    plt.barh(y_pos,          [r["medium"]*100 for r in rows],
             height=bar_h, color="#ffca28", label="Medium")
    plt.barh(y_pos - bar_h, [r["hard"]*100  for r in rows],
             height=bar_h, color="#d50000", label="Hard")

    # Achsen & Beschriftungen
    plt.yticks(y_pos, [r["model"] for r in rows])
    plt.xlim(0, 100)
    plt.xlabel(f"{metric_fn_name} (%)")
    plt.title(f"{metric_fn_name} nach Schwierigkeitsgrad")
    plt.legend(loc="lower right")

    # Balken beschriften
    for offset, key in [(bar_h, "easy"), (0, "medium"), (-bar_h, "hard")]:
        for y, r in zip(y_pos, rows):
            val = r[key] * 100
            plt.annotate(f"{val:.1f}%",
                         (val + 1, y + offset + bar_h / 4),
                         fontsize=8)

    plt.gca().invert_yaxis()
    plt.tight_layout()

def plot_superior(metrics: Dict[str, LLMResultMetrics], *, show_counts: bool = True) -> None:
    """
    Horizontal bar-chart showing *percentage* of superior judgments.
    Takes metrics dictionary as input.
    """
    # Calculate percentages and counts
    superior_data = {
        m.model_name: {
            'superior_cnt': len(m.superior_ids),
            'total': m.total_queries,
            'pct': (len(m.superior_ids) / m.total_queries * 100)
        }
        for model_name, m in metrics.items()
    }
    
    # Convert to DataFrame and sort by percentage
    df = pd.DataFrame.from_dict(superior_data, orient='index')
    pct = df['pct'].sort_values(ascending=False)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(pct) + 1))
    
    # Plot bars
    pct.plot.barh(
        ax=ax,
        color="#8e24aa",
    )
    
    # Set labels and limits
    ax.set_xlabel("Superior judgments (%)")
    ax.set_ylabel("model")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()  # best model on top

    # Annotate bars
    for bar, (model, pct_val) in zip(ax.patches, pct.items()):
        raw = superior_data[model]['superior_cnt']
        txt = f"{pct_val:.1f}%"
        if show_counts:
            txt += f" ({raw})"
        ax.annotate(
            txt,
            (bar.get_width() + 1, bar.get_y() + bar.get_height() / 4),
            fontsize=9,
        )

    plt.tight_layout()


def plot_difficulty_multi_metrics(metrics_dict: Dict[str, "LLMResultMetrics"]) -> None:
    """
    Vier Subplots (Intent / Tech / AST / Exact).
    Jeder Subplot: drei Balken (Easy / Medium / Hard) pro Modell.
    Modelle sind nach Intent-Accuracy (gesamt) absteigend sortiert.
    """
    # Farbpalette
    _EASY_COL, _MED_COL, _HARD_COL = "#00c853", "#ffca28", "#d50000"
    _METRIC_ORDER: List[str] = [
        "accuracy_judge_intent",
        "accuracy_judge_technical",
        "accuracy_ast",
        "accuracy_exact",
    ]
    _METRIC_TITLES = {
        "accuracy_judge_intent":      "Intent",
        "accuracy_judge_technical":   "Tech",
        "accuracy_ast":               "AST",
        "accuracy_exact":             "Exact",
    }
    # ------------------------------------------------------------------ #
    # 1) Daten einsammeln
    # ------------------------------------------------------------------ #
    rows = {}
    for name, m in metrics_dict.items():
        rows[name] = {
            "intent":     m.accuracy_judge_intent(),
            "tech":       m.accuracy_judge_technical(),
            "ast":        m.accuracy_ast(),
            "exact":      m.accuracy_exact(),
            "easy": {
                fn: getattr(m, fn)("easy") for fn in _METRIC_ORDER
            },
            "medium": {
                fn: getattr(m, fn)("medium") for fn in _METRIC_ORDER
            },
            "hard": {
                fn: getattr(m, fn)("hard") for fn in _METRIC_ORDER
            },
        }

    # Sortieren nach Intent-Accuracy
    model_order = sorted(rows.keys(), key=lambda k: rows[k]["intent"], reverse=True)
    n_models = len(model_order)
    y_pos = np.arange(n_models)
    bar_h = 0.22

    # ------------------------------------------------------------------ #
    # 2) Plot-Canvas
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(
        nrows=1, ncols=4,
        figsize=(15, 0.45 * n_models + 1),
        sharey=True
    )

    for ax, metric_fn in zip(axes, _METRIC_ORDER):
        # Easy / Medium / Hard Werte je Modell
        easy   = [rows[m]["easy"][metric_fn]   * 100 for m in model_order]
        medium = [rows[m]["medium"][metric_fn] * 100 for m in model_order]
        hard   = [rows[m]["hard"][metric_fn]   * 100 for m in model_order]

        # Drei Balken je Modell
        ax.barh(y_pos + bar_h, easy,   bar_h, color=_EASY_COL, label="Easy")
        ax.barh(y_pos,          medium, bar_h, color=_MED_COL,  label="Medium")
        ax.barh(y_pos - bar_h, hard,   bar_h, color=_HARD_COL,  label="Hard")

        ax.set_title(_METRIC_TITLES[metric_fn])
        ax.set_xlim(0, 100)
        ax.set_xlabel("%")
        ax.invert_yaxis()

        # Balkenbeschriftungen
        for idx, (e, m_, h) in enumerate(zip(easy, medium, hard)):
            for val, off in [(e, bar_h), (m_, 0), (h, -bar_h)]:
                if val > 0:
                    ax.annotate(f"{val:.1f}%",
                                (val + 1, idx + off + bar_h / 4),
                                fontsize=7)

        # y-Ticks nur links
        if ax is axes[0]:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(model_order)
            ax.legend(loc="lower right", fontsize=8)
        else:
            ax.set_yticks([])

    plt.tight_layout()

def plot_token_usage(metrics: Dict[str, LLMResultMetrics]) -> None:
    """
    Two side-by-side stacked bar charts showing prompt and completion token usage per model,
    separated by reasoning vs non-reasoning models.
    Models are ordered by total token usage (descending). Values shown in thousands (k).
    """
    # Split metrics into reasoning and non-reasoning
    reasoning_metrics = {k: v for k, v in metrics.items() if v.reasoning}
    non_reasoning_metrics = {k: v for k, v in metrics.items() if not v.reasoning}

    def create_and_sort_df(metrics_dict):
        # Extract token data
        token_data = {
            m.model_name: {
                'Prompt Tokens': m.total_prompt_tokens,
                'Completion Tokens': m.total_completion_tokens,
                'Total': m.total_prompt_tokens + m.total_completion_tokens
            }
            for model_name, m in metrics_dict.items()
        }
        
        # Convert to DataFrame and sort by total usage
        df = pd.DataFrame.from_dict(token_data, orient='index')
        return df.sort_values(by='Total', ascending=False).drop('Total', axis=1)

    # Create sorted DataFrames for both groups
    df_reasoning = create_and_sort_df(reasoning_metrics)
    df_non_reasoning = create_and_sort_df(non_reasoning_metrics)

    # Create side-by-side subplots
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, 
        figsize=(20, 0.4 * max(len(reasoning_metrics), len(non_reasoning_metrics)) + 1)
    )

    # Helper function to plot on each subplot
    def plot_on_axis(df, ax, title):
        df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=['#3498db', '#e74c3c']
        )
        
        ax.set_xlabel("Number of Tokens (thousands)")
        ax.set_ylabel("Model")
        ax.set_title(title)
        ax.invert_yaxis()

        # Add value labels with 'k' notation
        for c in ax.containers:
            labels = [f"{(v/1000):.1f}k" if v > 0 else "" for v in c.datavalues]
            ax.bar_label(
                c,
                labels=labels,
                label_type='center'
            )

    # Plot both subplots
    plot_on_axis(df_non_reasoning, ax1, "Non-Reasoning Models")
    plot_on_axis(df_reasoning, ax2, "Reasoning Models")

    # Adjust legend - only show once for both subplots
    handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(
    #     handles, 
    #     labels,
    #     bbox_to_anchor=(1.02, 0.5),
    #     loc='center left'
    # )

    plt.tight_layout()

def plot_model_comparison(metrics: Dict[str, LLMResultMetrics], 
                         accuracy_metric: str = 'intent',
                         save_path: Optional[str] = None,
                         custom_title: Optional[str] = None) -> None:
    """
    Create a scatter plot comparing models by accuracy, latency, and token usage.
    
    Parameters:
    -----------
    metrics : Dict[str, LLMResultMetrics]
        Dictionary of model metrics
    accuracy_metric : str
        One of 'exact', 'ast', 'technical', 'intent'
    save_path : Optional[str]
        Path to save the figure
    custom_title : Optional[str]
        Custom title for the plot
    """
    import matplotlib.patheffects as path_effects
    from adjustText import adjust_text
    import matplotlib.ticker as mticker
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")

    # Map accuracy metric to method
    accuracy_methods = {
        'exact': 'accuracy_exact',
        'ast': 'accuracy_ast',
        'technical': 'accuracy_judge_technical',
        'intent': 'accuracy_judge_intent'
    }
    accuracy_method = accuracy_methods.get(accuracy_metric.lower(), 'accuracy_judge_intent')

    # Create DataFrame from metrics
    model_data = []
    for model_name, m in metrics.items():
        accuracy_func = getattr(m, accuracy_method)
        model_data.append({
            'model': m.model_name,
            'accuracy': accuracy_func() * 100,
            'avg_time': m.avg_latency_sec,
            'avg_tokens': m.total_completion_tokens / m.total_queries if m.total_queries > 0 else 0,
            'open_source': m.open_source
        })

    df = pd.DataFrame(model_data)

    # Scale token counts for bubble sizes
    S_MIN, S_MAX = 100, 2000
    def scale_sizes(series, s_min=S_MIN, s_max=S_MAX):
        if len(series) <= 1 or series.max() == series.min():
            return np.full_like(series, (s_min + s_max) / 2)
        return np.interp(series, (series.min(), series.max()), (s_min, s_max))

    sizes = scale_sizes(df["avg_tokens"])

    # Define model categories and colors
    df['category'] = df['open_source'].apply(
        lambda x: 'Open Source Models' if x else 'Closed Source Models')
    category_colors = {
        'Open Source Models': '#009E73',
        'Closed Source Models': '#0072B2'
    }
    df['color'] = df['category'].map(category_colors)

    # Create plot
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xscale('log')
    ax.grid(which='major', linestyle='--', alpha=0.7, zorder=0)
    ax.grid(which='minor', linestyle=':', alpha=0.4, zorder=0)

    # Create scatter plot
    scatter = ax.scatter(
        df["avg_time"], df["accuracy"], s=sizes, c=df["color"],
        alpha=0.8, edgecolors='black', linewidth=1.2, zorder=10
    )

    # Add model labels
    texts = []
    for _, row in df.iterrows():
        text = ax.text(
            row.avg_time * 1.05, row.accuracy, row.model,
            fontsize=11, fontweight='bold', zorder=25,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7, 
                     pad=1.5, boxstyle='round,pad=0.5')
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white', alpha=0.8),
            path_effects.Normal()
        ])
        texts.append(text)

    # Adjust text positions to avoid overlap
    if len(texts) > 1:
        try:
            adjust_text(texts, ax=ax,
                       force_text=(10.5, 10.5),
                       force_points=(10.2, 10.2),
                       expand_text=(1.1, 1.1),
                       expand_points=(1.05, 1.05),
                       arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
                       autoalign=False,
                       lim=20)
        except Exception as e:
            print(f"adjust_text error: {e}. Continuing without adjusting text.")

    # Add legends
    # Model category legend
    category_handles = [
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=color, markeredgecolor='black',
               markersize=15, label=category)
        for category, color in category_colors.items()
    ]
    type_legend = ax.legend(handles=category_handles, loc="upper right",
                           fontsize=14, frameon=True, framealpha=0.95,
                           title="Legend", title_fontsize=16)
    ax.add_artist(type_legend)

    # Token size legend
    if len(df) > 1:
        token_vals = [
            int(df["avg_tokens"].min()),
            int(df["avg_tokens"].median()),
            int(df["avg_tokens"].max())
        ]
        size_vals = scale_sizes(pd.Series(token_vals))
        handles = [plt.scatter([], [], s=s, color="gray", edgecolor="black", linewidth=1.0)
                  for s in size_vals]
        labels = [f"{t:,} tokens" for t in token_vals]
        size_legend = ax.legend(handles, labels, title="Avg Completion Tokens",
                              loc="lower right", fontsize=14, title_fontsize=16,
                              labelspacing=2.2, borderpad=1.2, handletextpad=1.5,
                              frameon=True, framealpha=0.95)
        ax.add_artist(size_legend)

    # Labels and title
    ax.set_xlabel("Average Query Time (seconds, log scale) →", 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel("↑ Accuracy (%)", fontsize=16, fontweight='bold')

    metric_display_names = {
        'exact': 'Exact Match',
        'ast': 'AST Match',
        'technical': 'Technical Accuracy',
        'intent': 'Intent Fulfillment'
    }
    metric_display = metric_display_names.get(accuracy_metric.lower(), 
                                            accuracy_metric.title())

    title = (custom_title if custom_title else
            f"SQL Model Performance: {metric_display} vs. Latency")
    ax.set_title(title, fontsize=20, fontweight='bold', pad=25)

    # Configure axes
    major_ticks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    # Set axis limits
    data_min = max(0, df["accuracy"].min() - 5)
    data_max = min(100, df["accuracy"].max() + 5)
    ax.set_ylim(data_min, data_max)
    ax.set_xlim(left=0.8, right=df["avg_time"].max() * 1.2)

    # Add note about metric
    note_texts = {
        'exact': "Note: Accuracy based on exact string match.",
        'ast': "Note: Accuracy based on Abstract Syntax Tree matching.",
        'technical': "Note: Accuracy based on technical equivalence assessment.",
        'intent': "Note: Accuracy based on intent fulfillment assessment."
    }
    note_text = note_texts.get(accuracy_metric.lower(), 
                             f"Note: Accuracy based on '{accuracy_metric}' assessment.")
    plt.figtext(0.5, 0.01, note_text, ha='center', fontsize=12, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

def plot_complexity_heatmap(metrics: Dict[str, LLMResultMetrics], complexity_map: Dict[int, str]) -> None:
    """
    Create a heatmap showing the relationship between SQL complexity and difficulty.
    Uses any model's metrics as they all have the same distribution.
    """
    # Take first model's metrics as they all have the same distribution
    any_model = next(iter(metrics.values()))
    
    # Get SQL complexity counts
    sql_complexities = list(any_model.per_sql_complexity_counts.keys())
    
    # Create mapping data
    data = defaultdict(lambda: defaultdict(int))
    
    # For each SQL complexity type, count examples in each difficulty level
    for sql_type in sql_complexities:
        example_ids = any_model._sqlclass_id_map[sql_type]
        for eid in example_ids:
            difficulty = complexity_map.get(eid, "unknown").lower()
            data[sql_type][difficulty] += 1
    
    # Convert to DataFrame and ensure integer values
    df = pd.DataFrame.from_dict(data, orient='index').fillna(0).astype(int)
    
    # Ensure all difficulty levels are present and in correct order
    difficulty_order = ["easy", "medium", "hard"]
    for diff in difficulty_order:
        if diff not in df.columns:
            df[diff] = 0
    df = df[difficulty_order]
    
    # Define desired SQL complexity order
    sql_order = [
        "basic SQL",
        "aggregation",
        "single join",
        "set operations",
        "subqueries",
        "multiple_joins",
        "window functions"
    ]
    
    # Reorder the DataFrame and fill any missing SQL types with 0
    df = df.reindex(sql_order).fillna(0).astype(int)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with adjusted text properties
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={'label': 'Number of Questions'},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    # Customize plot
    plt.title("SQL Complexity vs Difficulty Distribution (196 Examples)", 
              fontsize=14, pad=20)
    plt.ylabel("SQL Complexity Type (28 Examples each)", fontsize=12)
    plt.xlabel("Difficulty", fontsize=12)
    
    # --- MODIFICATION START ---
    # Calculate column sums for difficulty levels
    column_sums = df.sum(axis=0)
    
    # Create new x-tick labels with totals
    new_xticklabels = [f"{col.capitalize()}\n({(column_sums[col] / len(complexity_map)) * 100:.1f} %)" for col in df.columns]
    
    # Apply the new labels to the plot
    ax = plt.gca()
    ax.set_xticklabels(new_xticklabels, rotation=0, ha="center")
    # --- MODIFICATION END ---
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=3.0) # Added padding to ensure new labels fit
    
    plt.show()