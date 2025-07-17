# viz_toolbox.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from typing import Dict, List

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
        model_name: {
            "exact": m.accuracy_exact() * 100,
            "ast": m.accuracy_ast() * 100,
            "technical": m.accuracy_judge_technical() * 100,
            "intent": m.accuracy_judge_intent() * 100
        }
        for model_name, m in metrics.items()
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
                model=name,
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
        model_name: {
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
            model_name: {
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