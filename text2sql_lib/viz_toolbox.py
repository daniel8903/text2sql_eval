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
# Define color scheme at the top - Change these values to update all plots
COLOR_SCHEME = {
    # Primary colors for single-color plots
    "primary": "#3498db",  # Grün
    "secondary": "#f1c40f", # Gelb
    "tertiary": "#e74c3c",  # Rot
    
    # Colors for different metrics - wie gewünscht
    "intent": "#2ecc71",    # Grün
    "technical": "#3498db", # Blau
    "ast": "#e74c3c",       # Rot
    "exact": "#f1c40f",     # Gelb
    
    # Colors for difficulty levels - passend zu den Metrikfarben
    "easy": "#2ecc71",      # Grün (wie intent)
    "medium": "#f1c40f",    # Gelb (wie exact)
    "hard": "#e74c3c",      # Rot (wie ast)
    
    # Colors for model categories - konsistent mit Primärfarben
    "open_source": "#2ecc71",    # Grün
    "closed_source": "#3498db",  # Blau
    
    # Colors for token types - konsistent mit Primärfarben
    "prompt_tokens": "#3498db",  # Blau
    "completion_tokens": "#e74c3c", # Rot
    
    # Colormap for heatmaps - passend zum Farbschema
    "heatmap": "Blues"  # Rot-Gelb-Grün Farbverlauf (reversed)
}

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

    # Use color scheme from the top
    _CMAP = {
        "intent": COLOR_SCHEME["intent"],
        "technical": COLOR_SCHEME["technical"],
        "ast": COLOR_SCHEME["ast"],
        "exact": COLOR_SCHEME["exact"]
    }

    # Plot each metric
    for ax, metric in zip(axes, metric_cols):
        df_sorted[metric].plot.barh(
            ax=ax,
            color=_CMAP[metric],
        )
        ax.set_title(metric.capitalize())
        ax.set_xlim(0, 100)
        ax.set_xlabel("Genauigkeit %")
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
    axes[0].set_ylabel("LLM")
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

    # Use color scheme from the top
    _EASY_COL = COLOR_SCHEME["easy"]
    _MED_COL = COLOR_SCHEME["medium"]
    _HARD_COL = COLOR_SCHEME["hard"]

    plt.figure(figsize=(10, 0.6 * len(rows) + 1))

    # 3 Balken je Modell
    plt.barh(y_pos + bar_h, [r["easy"]*100  for r in rows],
             height=bar_h, color=_EASY_COL, label="Einfach")
    plt.barh(y_pos,          [r["medium"]*100 for r in rows],
             height=bar_h, color=_MED_COL, label="Mittel")
    plt.barh(y_pos - bar_h, [r["hard"]*100  for r in rows],
             height=bar_h, color=_HARD_COL, label="Schwer")
    
    SHORT_METRIC_TITLE_MAP = {
        "accuracy_exact": "Exakte Übereinstimmung",
        "accuracy_ast": "Syntaxbaum-Übereinstimmung",
        "accuracy_judge_technical": "Technische Korrektheit",
        "accuracy_judge_intent": "Intentionserfüllung"
    }

    # Achsen & Beschriftungen
    plt.yticks(y_pos, [r["model"] for r in rows])
    plt.xlim(0, 100)
    plt.xlabel(f"{metric_fn_name} (%)")
    plt.title(f"{SHORT_METRIC_TITLE_MAP.get(metric_fn_name, metric_fn_name)} nach Schwierigkeitsgrad", fontsize=14)
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
    
    # Plot bars with color from color scheme
    pct.plot.barh(
        ax=ax,
        color=COLOR_SCHEME["primary"],
    )
    
    # Set labels and limits
    ax.set_xlabel("Überlegene Beurteilungen (%)")
    ax.set_ylabel("LLM")
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

def plot_superior_horizontal(metrics: Dict[str, LLMResultMetrics], *, show_counts: bool = True) -> None:
    """
    Vertikales Balkendiagramm der *prozentualen* überlegenen Bewertungen.
    Nutzt ein Dictionary mit Metriken als Eingabe.
    """
    # Daten berechnen: Prozentsätze und absolute Werte
    superior_data = {
        m.model_name: {
            'superior_cnt': len(m.superior_ids),
            'total': m.total_queries,
            'pct': (len(m.superior_ids) / m.total_queries * 100)
        }
        for model_name, m in metrics.items()
    }

    # In DataFrame umwandeln und sortieren
    df = pd.DataFrame.from_dict(superior_data, orient='index')
    pct = df['pct'].sort_values(ascending=False)

    # Plot-Vorbereitung
    fig, ax = plt.subplots(figsize=(0.6 * len(pct) + 2, 5))  # Breite anpassen an Modellanzahl

    # Balkenplot (vertikal)
    pct.plot.bar(
        ax=ax,
        color=COLOR_SCHEME["primary"],
        width=0.6
    )

    # Achsenbeschriftungen und Limits
    ax.set_ylabel("Überlegene Beurteilungen (%)")
    ax.set_xlabel("LLM")
    ax.set_ylim(0, 100)
    ax.set_xticklabels(pct.index, rotation=45, ha='right')

    # Balkenbeschriftungen hinzufügen
    for bar, (model, pct_val) in zip(ax.patches, pct.items()):
        raw = superior_data[model]['superior_cnt']
        txt = f"{pct_val:.1f}%"
        if show_counts:
            txt += f"\n({raw})"
        ax.annotate(
            txt,
            (bar.get_x() + bar.get_width() / 2, bar.get_height() + 1),
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()


def plot_all_metrics(metrics_dict: Dict[str, "LLMResultMetrics"]) -> None:
    """
    Erstellt einen Plot mit 4 Subplots für alle Metriken aus SHORT_METRIC_TITLE_MAP
    """
    SHORT_METRIC_TITLE_MAP = {
        "accuracy_exact": "Exakte Übereinstimmung",
        "accuracy_ast": "Syntaxbaum-Übereinstimmung",
        "accuracy_judge_technical": "Technische Korrektheit",
        "accuracy_judge_intent": "Intentionserfüllung"
    }
    
    # Farbschema definieren (falls COLOR_SCHEME nicht verfügbar ist)
    color_scheme = {
        "easy": "#8BC34A",    # Grün
        "medium": "#FFC107",  # Gelb
        "hard": "#FF5722",    # Orange
        "primary": "#2196F3"  # Blau
    }
    
    # Figur mit 2x2 Subplots erstellen
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()  # Für einfacheren Zugriff
    
    # Für jede Metrik einen Subplot erstellen
    for i, (metric_fn_name, title) in enumerate(SHORT_METRIC_TITLE_MAP.items()):
        ax = axes[i]
        
        # 1) Daten sammeln
        rows = []
        for name, m in metrics_dict.items():
            fn = getattr(m, metric_fn_name)
            rows.append(
                dict(
                    model=m.model_name,
                    total=fn(),
                    easy=fn("easy"),
                    medium=fn("medium"),
                    hard=fn("hard"),
                )
            )
        
        # Sortieren nach Gesamtgenauigkeit
        rows.sort(key=lambda r: r["total"], reverse=True)
        
        # 2) Plot vorbereiten
        y_pos = np.arange(len(rows))
        bar_h = 0.25
        
        # Farben definieren
        _EASY_COL = color_scheme["easy"]
        _MED_COL = color_scheme["medium"]
        _HARD_COL = color_scheme["hard"]
        
        # 3 Balken je Modell
        ax.barh(y_pos + bar_h, [r["easy"]*100 for r in rows],
                height=bar_h, color=_EASY_COL, label="Einfach")
        ax.barh(y_pos, [r["medium"]*100 for r in rows],
                height=bar_h, color=_MED_COL, label="Mittel")
        ax.barh(y_pos - bar_h, [r["hard"]*100 for r in rows],
                height=bar_h, color=_HARD_COL, label="Schwer")
        
        # Achsen & Beschriftungen
        ax.set_yticks(y_pos)
        ax.set_yticklabels([r["model"] for r in rows])
        ax.set_xlim(0, 100)
        ax.set_xlabel(f"Genauigkeit (%)")
        ax.set_title(f"{title} nach Schwierigkeitsgrad", fontsize=14)
        
        # Nur beim ersten Subplot Legende anzeigen
        if i == 0:
            ax.legend(loc="lower right")
        
        # Balken beschriften
        for offset, key in [(bar_h, "easy"), (0, "medium"), (-bar_h, "hard")]:
            for y, r in zip(y_pos, rows):
                val = r[key] * 100
                ax.annotate(f"{val:.1f}%",
                           (val + 1, y + offset + bar_h / 4),
                           fontsize=8)
        
        ax.invert_yaxis()  # Bestes Modell oben
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle("Vergleich aller Genauigkeitsmetriken", fontsize=16, y=0.98)
    plt.show()

def plot_all_metrics_horizontal_grouped(metrics_dict: Dict[str, "LLMResultMetrics"]) -> None:
    """
    Erstellt einen Plot mit 4 nebeneinander angeordneten Subplots für alle Metriken,
    wobei jeder Balken nach Schwierigkeitsgraden (easy, medium, hard) aufgeteilt ist.
    """
    # Metriken in der gewünschten Reihenfolge
    metrics_order = [
        "accuracy_judge_intent",
        "accuracy_judge_technical", 
        "accuracy_ast", 
        "accuracy_exact"
    ]
    
    # Kurze Titel für die Metriken
    metric_titles = {
        "accuracy_judge_intent": "Intent",
        "accuracy_judge_technical": "Technical",
        "accuracy_ast": "Ast",
        "accuracy_exact": "Exact"
    }
    
    # Sammle alle Modellnamen und sortiere nach Intent-Metrik
    all_models = []
    for name, m in metrics_dict.items():
        intent_fn = getattr(m, "accuracy_judge_intent")
        all_models.append((m.model_name, intent_fn()))
    
    # Sortiere Modelle nach Intent-Genauigkeit (absteigend)
    all_models.sort(key=lambda x: x[1], reverse=False)
    model_order = [m[0] for m in all_models]
    
    # Figur erstellen - mehr Platz für die Modellnamen
    fig, axes = plt.subplots(1, 4, figsize=(20, 12), sharey=True)
    
    # Y-Position für alle Modelle
    y_pos = np.arange(len(model_order))
    
    # Für jede Metrik einen Subplot erstellen
    for i, metric_fn_name in enumerate(metrics_order):
        ax = axes[i]
        
        # Daten für easy, medium und hard sammeln
        easy_values = []
        medium_values = []
        hard_values = []
        
        for model_name in model_order:
            m = next(m for name, m in metrics_dict.items() if m.model_name == model_name)
            fn = getattr(m, metric_fn_name)
            easy_values.append(fn("easy") * 100)
            medium_values.append(fn("medium") * 100)
            hard_values.append(fn("hard") * 100)
        
        # Balken für jeden Schwierigkeitsgrad zeichnen
        bar_height = 0.7
        small_bar_height = bar_height / 3
        
        # Hard (unten)
        hard_bars = ax.barh(y_pos - small_bar_height, hard_values, 
                           color=COLOR_SCHEME["hard"], 
                           height=small_bar_height, 
                           label="Schwer")
        
        # Medium (mitte)
        medium_bars = ax.barh(y_pos, medium_values, 
                             color=COLOR_SCHEME["medium"], 
                             height=small_bar_height, 
                             label="Mittel")
        
        # Easy (oben)
        easy_bars = ax.barh(y_pos + small_bar_height, easy_values, 
                           color=COLOR_SCHEME["easy"], 
                           height=small_bar_height, 
                           label="Einfach")
        
        # Werte an den Balken anzeigen - nur wenn sie größer als ein Schwellwert sind
        threshold = 5.0  # Nur Werte > 5% anzeigen
        
        for j in range(len(y_pos)):
            # Hard
            if hard_values[j] > threshold:
                ax.text(hard_values[j] + 0.5, y_pos[j] - small_bar_height, 
                       f"{hard_values[j]:.1f}%", va='center', fontsize=7)
            # Medium
            if medium_values[j] > threshold:
                ax.text(medium_values[j] + 0.5, y_pos[j], 
                       f"{medium_values[j]:.1f}%", va='center', fontsize=7)
            # Easy
            if easy_values[j] > threshold:
                ax.text(easy_values[j] + 0.5, y_pos[j] + small_bar_height, 
                       f"{easy_values[j]:.1f}%", va='center', fontsize=7)
        
        # Achsen & Beschriftungen
        ax.set_title(metric_titles[metric_fn_name], fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Genauigkeit %", fontsize=10)
        
        # Gitter nur für x-Achse
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        if i == 0:
            ax.set_yticks(y_pos)
            ax.set_yticklabels(model_order, fontsize=9)
            ax.set_ylabel("LLM", fontsize=10)
        else:
            # keine Änderung an den (geteilten) Labels vornehmen!
            ax.tick_params(axis="y", labelleft=False)   # nur ausblenden
    
    # Gemeinsame Legende unten
    fig.legend(
        [easy_bars, medium_bars, hard_bars],
        ["Einfach", "Mittel", "Schwer"],
        loc='lower center',
        ncol=3,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=10
    )
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, bottom=0.08, left=0.15)
    
    return fig

def plot_difficulty_multi_metrics(metrics_dict: Dict[str, "LLMResultMetrics"], 
                                 complexity_map: Dict[int, str],
                                 sort_by: str = "intent") -> None:
    """
    Vier Subplots (Intent / Tech / AST / Exact).
    Jeder Subplot: drei Balken (Einfach / Mittel / Schwer) pro LLM.
    """
    # Farbschema
    _EASY_COL = COLOR_SCHEME["easy"]
    _MED_COL = COLOR_SCHEME["medium"]
    _HARD_COL = COLOR_SCHEME["hard"]
    
    _METRIC_ORDER: List[str] = [
        "accuracy_judge_intent",
        "accuracy_judge_technical",
        "accuracy_ast",
        "accuracy_exact",
    ]
    
    _METRIC_TITLES = {
        "accuracy_judge_intent":      "Intentionserfüllung",
        "accuracy_judge_technical":   "Technische Korrektheit",
        "accuracy_ast":               "AST-Übereinstimmung",
        "accuracy_exact":             "Exakte Übereinstimmung",
    }
    
    # ------------------------------------------------------------------ #
    # 1) Daten einsammeln
    # ------------------------------------------------------------------ #
    rows = {}
    for name, m in metrics_dict.items():
        rows[name] = {
            "intent":     m.accuracy_judge_intent(),
            "technical":  m.accuracy_judge_technical(),
            "ast":        m.accuracy_ast(),
            "exact":      m.accuracy_exact(),
            "average":    (m.accuracy_judge_intent() + 
                          m.accuracy_judge_technical() + 
                          m.accuracy_ast() + 
                          m.accuracy_exact()) / 4,
            "combined":   m.accuracy_judge_intent() * 0.4 + 
                          m.accuracy_judge_technical() * 0.3 + 
                          m.accuracy_ast() * 0.2 + 
                          m.accuracy_exact() * 0.1,
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

    valid_sort_criteria = ["intent", "technical", "ast", "exact", "average", "combined"]
    if sort_by not in valid_sort_criteria:
        sort_by = "intent"
    
    model_order = sorted(rows.keys(), key=lambda k: rows[k][sort_by], reverse=True)
    n_models = len(model_order)
    y_pos = np.arange(n_models)
    bar_h = 0.22

    # ------------------------------------------------------------------ #
    # 2) KOMPLETT NEUER ANSATZ: Figur mit zwei Gridspecs
    # ------------------------------------------------------------------ #
    fig = plt.figure(figsize=(20, max(10, 0.7 * n_models)))
    
    # Zwei separate GridSpecs:
    # 1. Für Modellnamen (schmal, links)
    # 2. Für die vier Metrikplots (breit, rechts)
    gs_names = fig.add_gridspec(1, 1, left=0.01, right=0.2, wspace=0)
    gs_plots = fig.add_gridspec(1, 4, left=0.21, right=0.99, wspace=0.05)
    
    # Namen-Subplot
    ax_names = fig.add_subplot(gs_names[0, 0])
    ax_names.set_axis_off()  # Achsen ausblenden
    
    # Vier Metrik-Subplots
    axes = [fig.add_subplot(gs_plots[0, i]) for i in range(4)]
    
    # ------------------------------------------------------------------ #
    # 3) Daten plotten
    # ------------------------------------------------------------------ #
    for i, (ax, metric_fn) in enumerate(zip(axes, _METRIC_ORDER)):
        # Werte für alle Modelle sammeln
        easy   = [rows[m]["easy"][metric_fn]   * 100 for m in model_order]
        medium = [rows[m]["medium"][metric_fn] * 100 for m in model_order]
        hard   = [rows[m]["hard"][metric_fn]   * 100 for m in model_order]

        # Balken plotten
        ax.barh(y_pos + bar_h, easy,   bar_h, color=_EASY_COL, label="Einfach" if i == 0 else "")
        ax.barh(y_pos,          medium, bar_h, color=_MED_COL,  label="Mittel" if i == 0 else "")
        ax.barh(y_pos - bar_h, hard,   bar_h, color=_HARD_COL,  label="Schwer" if i == 0 else "")

        ax.set_title(_METRIC_TITLES[metric_fn], fontsize=13)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Genauigkeit (%)", fontsize=11)
        
        # Leere Y-Achse bei allen Plots
        ax.set_yticks([])
        ax.invert_yaxis()  # Umkehren, damit die besten Modelle oben sind

        # Balkenbeschriftungen
        for idx, (e, m_, h) in enumerate(zip(easy, medium, hard)):
            for val, off in [(e, bar_h), (m_, 0), (h, -bar_h)]:
                if val > 0:
                    ax.annotate(f"{val:.1f}%",
                                (val + 1, idx + off + bar_h / 4),
                                fontsize=7)
    
    # ------------------------------------------------------------------ #
    # 4) LLM-Namen links plotten
    # ------------------------------------------------------------------ #
    # Modellnamen manuell links platzieren
    for i, model in enumerate(model_order):
        # Debug-Output, um zu sehen was in model_order steht
        print(f"Platziere Modellnamen {i+1}: {model}")
        
        # Text mit Rangzahl
        ax_names.text(0.95, y_pos[i], f"{i+1}. {model}", 
                     ha='right', va='center',
                     fontsize=10, fontweight='bold')
    
    # Titel für die Namensspalte
    ax_names.text(0.5, -1, "LLM-Ranking", 
                 ha='center', va='center',
                 fontsize=12, fontweight='bold')
    
    # ------------------------------------------------------------------ #
    # 5) Legende und Y-Achsen-Limits anpassen
    # ------------------------------------------------------------------ #
    # Alle Plots auf gleichen Y-Bereich setzen
    for ax in axes:
        ax.set_ylim(min(y_pos) - 1, max(y_pos) + 1)
    
    # Legende für den ersten Plot
    axes[0].legend(loc="lower right", fontsize=9)
    
    # ------------------------------------------------------------------ #
    # 6) Titel und Anmerkungen
    # ------------------------------------------------------------------ #
    sort_title_map = {
        "intent": "Intentionserfüllung",
        "technical": "technische Korrektheit",
        "ast": "AST-Übereinstimmung",
        "exact": "exakte Übereinstimmung",
        "average": "Durchschnitt aller Metriken",
        "combined": "gewichteter Durchschnitt (40/30/20/10)"
    }
    
    sort_title = sort_title_map.get(sort_by, sort_by)
    plt.suptitle(f"LLM-Performance nach Schwierigkeitsgrad\n(sortiert nach {sort_title})", 
                 fontsize=15, y=0.98)
    
    # Anzahl der Fragen pro Schwierigkeitsgrad
    easy_count = len([eid for eid in range(1, 73) if complexity_map.get(eid) == 'easy'])
    medium_count = len([eid for eid in range(1, 73) if complexity_map.get(eid) == 'medium'])
    hard_count = len([eid for eid in range(1, 73) if complexity_map.get(eid) == 'hard'])
    
    plt.figtext(0.5, 0.01, 
               f"Anzahl der Fragen: Einfach ({easy_count}), Mittel ({medium_count}), Schwer ({hard_count})",
               ha="center", fontsize=9)

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

    # Define model categories and colors - use colors from color scheme
    df['category'] = df['open_source'].apply(
        lambda x: 'Open Source LLMs' if x else 'Closed Source LLMs')
    category_colors = {
        'Open Source LLMs': COLOR_SCHEME["open_source"],
        'Closed Source LLMs': COLOR_SCHEME["closed_source"]
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
                           title="Legende", title_fontsize=16)
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
        labels = [f"{t:,} Tokens" for t in token_vals]
        size_legend = ax.legend(handles, labels, title="Ø Completion-Tokens",
                              loc="lower right", fontsize=14, title_fontsize=16,
                              labelspacing=2.2, borderpad=1.2, handletextpad=1.5,
                              frameon=True, framealpha=0.95)
        ax.add_artist(size_legend)

    # Labels and title
    ax.set_xlabel("Ø Anfragezeit (Sekunden, logarithmische Skala) →", 
                 fontsize=16, fontweight='bold')
    ax.set_ylabel("↑ Genauigkeit (%)", fontsize=16, fontweight='bold')

    metric_display_names = {
        'exact': 'Exakte Übereinstimmung',
        'ast': 'AST-Übereinstimmung',
        'technical': 'Technische Genauigkeit',
        'intent': 'Intentionserfüllung'
    }
    metric_display = metric_display_names.get(accuracy_metric.lower(), 
                                            accuracy_metric.title())

    title = (custom_title if custom_title else
            f"SQL-LLM-Leistung: {metric_display} vs. Latenz")
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
        'exact': "Hinweis: Genauigkeit basiert auf exakter String-Übereinstimmung.",
        'ast': "Hinweis: Genauigkeit basiert auf Abstract Syntax Tree-Übereinstimmung.",
        'technical': "Hinweis: Genauigkeit basiert auf technischer Äquivalenzbewertung.",
        'intent': "Hinweis: Genauigkeit basiert auf Intentionserfüllungsbewertung."
    }
    note_text = note_texts.get(accuracy_metric.lower(), 
                             f"Hinweis: Genauigkeit basiert auf '{accuracy_metric}'-Bewertung.")
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
    
    # Create heatmap with color scheme from the top
    sns.heatmap(
        df,
        annot=True,
        fmt="d",
        cmap=COLOR_SCHEME["heatmap"],  # Using the heatmap colormap from color scheme
        cbar_kws={'label': 'Anzahl der Fragen'},
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    
    # Customize plot
    plt.title("SQL-Komplexität vs. Schwierigkeitsverteilung (196 Beispiele)", 
              fontsize=14, pad=20)
    plt.ylabel("SQL-Komplexitätstyp (28 Beispiele je Typ)", fontsize=12)
    plt.xlabel("Schwierigkeitsgrad", fontsize=12)
    
    # Calculate column sums for difficulty levels
    column_sums = df.sum(axis=0)

    # Vor dem f-string definieren
    difficulty_translation = {'easy': 'Einfach', 'medium': 'Mittel', 'hard': 'Schwer'}

    # Dann im f-string verwenden
    new_xticklabels = [f"{difficulty_translation[col]}\n({(column_sums[col] / len(complexity_map)) * 100:.1f} %)" for col in df.columns]
    
    # Create new x-tick labels with totals
    # new_xticklabels = [f"{{'easy': 'Einfach', 'medium': 'Mittel', 'hard': 'Schwer'}[col]}\n({(column_sums[col] / len(complexity_map)) * 100:.1f} %)" for col in df.columns]
    
    # Apply the new labels to the plot
    ax = plt.gca()
    ax.set_xticklabels(new_xticklabels, rotation=0, ha="center")
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout(pad=3.0)
    
    plt.show()

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
    fig.suptitle("Token-Verbrauch nach Modell-Typ (insgesamt 196 Anfragen)", fontsize=16)

    # Helper function to plot on each subplot
    def plot_on_axis(df, ax, title):
        df.plot(
            kind='barh',
            stacked=True,
            ax=ax,
            color=['#3498db', '#e74c3c']
        )
        
        ax.set_xlabel("Anzahl Tokens (in Tausend)")
        ax.set_ylabel("Model")
        ax.set_title(title)
        ax.invert_yaxis()

        # X-Achsen-Ticks ebenfalls in 'k' formatieren
        ax.xaxis.set_major_formatter(lambda x, pos: f'{x/1000:.0f}k')
        ax.legend(title='Token Typ', loc='lower right')

        # Add value labels with 'k' notation
        for c in ax.containers:
            labels = [f"{(v/1000):.1f}k" if v > 0 else "" for v in c.datavalues]
            ax.bar_label(
                c,
                labels=labels,
                label_type='center'
            )
    plt.title("Token-Verbrauch nach Modell (insgesamt 196 Beispiele)")

    # Plot both subplots
    plot_on_axis(df_non_reasoning, ax1, "Non-Reasoning Modelle")
    plot_on_axis(df_reasoning, ax2, "Reasoning Modelle")

    # Adjust legend - only show once for both subplots
    handles, labels = ax2.get_legend_handles_labels()
    # fig.legend(
    #     handles, 
    #     labels,
    #     bbox_to_anchor=(1.02, 0.5),
    #     loc='center left'
    # )

    plt.tight_layout()

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from adjustText import adjust_text

def create_model_comparison_chart(
    metrics_dict, 
    accuracy_metric='intent',  # 'exact', 'ast', 'technical', or 'intent'
    save_path=None
):
    """
    Create a scatter plot comparing models by accuracy, latency, and token usage.
    """
    # Set style for a cleaner look
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")

    # Map accuracy metric to method
    accuracy_method = {
        'exact': 'accuracy_exact',
        'ast': 'accuracy_ast',
        'technical': 'accuracy_judge_technical',
        'intent': 'accuracy_judge_intent'
    }.get(accuracy_metric.lower(), 'accuracy_judge_intent')
    
    # Create DataFrame from metrics
    model_data = []
    for model_key, m in metrics_dict.items():
        accuracy_func = getattr(m, accuracy_method)
        model_data.append({
            'model': m.model_name,
            'accuracy': accuracy_func() * 100,  # Convert to percentage
            'avg_time': m.avg_latency_sec,
            'avg_tokens': m.total_completion_tokens / m.total_queries if m.total_queries > 0 else 0,
            'total_examples': m.total_queries,
            'open_source': m.open_source
        })

    acc_df = pd.DataFrame(model_data)
    
    # Define model categories and colors using COLOR_SCHEME
    acc_df['category'] = acc_df['open_source'].apply(
        lambda x: 'Open Source Models' if x else 'Closed Source Models')
    category_colors = {
        'Open Source Models': COLOR_SCHEME["open_source"],
        'Closed Source Models': COLOR_SCHEME["closed_source"]
    }
    acc_df['color'] = acc_df['category'].map(category_colors).fillna('#999999')
    
    # Scale token counts for bubble sizes
    S_MIN, S_MAX = 100, 2000
    def scale_sizes(series, s_min=S_MIN, s_max=S_MAX):
        if len(series) <= 1 or series.max() == series.min():
            return np.full_like(series, (s_min + s_max) / 2)
        return np.interp(series, (series.min(), series.max()), (s_min, s_max))
        
    sizes = scale_sizes(acc_df["avg_tokens"])

    # Create plot with log scale
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xscale('log')
    ax.grid(which='major', linestyle='--', alpha=0.7, zorder=0)
    ax.grid(which='minor', linestyle=':', alpha=0.4, zorder=0)
    
    # # Add "efficient & accurate" zone using COLOR_SCHEME
    # ax.axvspan(0, 6, ymin=0.75, facecolor=COLOR_SCHEME["intent"], alpha=0.05, zorder=1)
    # ax.text(2.5, 97, 'Efficient & Accurate Zone\n(>90% Accuracy, <6s Latency)',
    #         fontsize=14, color=COLOR_SCHEME["intent"], style='italic', weight='bold')
    
    # Create scatter plot
    scatter = ax.scatter(
        acc_df["avg_time"], acc_df["accuracy"], s=sizes, c=acc_df["color"],
        alpha=0.8, edgecolors='black', linewidth=1.2, zorder=10
    )

    # Add model labels with adjustText
    texts = []
    for i, row in acc_df.iterrows():
        text = ax.text(
            row.avg_time, row.accuracy, row.model,
            fontsize=11 if row['avg_time'] > 10 else 13,
            fontweight='bold', zorder=25,
            color='black', # Keeping black for text for readability
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.5, pad=1.5)
        )
        text.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground='white', alpha=0.8),
            path_effects.Normal()
        ])
        texts.append(text)

    # Use strong force settings to prevent label overlap
    if len(texts) > 1:
        adjust_text(texts, ax=ax,
                force_text=(10.0, 10.0),
                force_points=(10.0, 10.0),
                expand_text=(5, 5),
                expand_points=(10.5, 10.5),
                lim=1000)

    # Add legend for model categories
    category_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                            markeredgecolor='black', markersize=15, label=category) 
                        for category, color in category_colors.items()]
    type_legend = ax.legend(handles=category_handles, loc="upper right", fontsize=14, 
                            frameon=True, framealpha=0.95, title="Modell Typ", title_fontsize=16)
    ax.add_artist(type_legend)

    # Create token size legend
    if len(acc_df) > 1:
        token_vals = [int(acc_df["avg_tokens"].min()), 
                    int(acc_df["avg_tokens"].median()), 
                    int(acc_df["avg_tokens"].max())]
        size_vals = scale_sizes(pd.Series(token_vals))
        handles = [plt.scatter([], [], s=s, color="gray", edgecolor="black", linewidth=1.0) # Keeping gray for token legend
                  for s in size_vals]
        labels = [f"{t:,} tokens" for t in token_vals]
        size_legend = ax.legend(handles, labels, title="Ø Completion Tokens", 
                              loc="lower right", fontsize=14, title_fontsize=16, 
                              labelspacing=2.2, borderpad=1.2, handletextpad=1.5, 
                              frameon=True, framealpha=0.95)
        ax.add_artist(size_legend)

    # Set labels and title
    ax.set_xlabel("Ø Latenz (Sekunden, Log Skala)", fontsize=16, fontweight='bold')
    ax.set_ylabel("Accuracy (%)", fontsize=16, fontweight='bold')
    ax.set_title(f"SQL Modell Performance Vergleich: {accuracy_metric.title()} Accuracy vs. Latenz", 
                fontsize=20, fontweight='bold', pad=25)

    # Configure axis ticks and limits
    major_ticks = [1, 2, 3, 5, 7, 10, 15, 20, 30, 40]
    ax.xaxis.set_major_locator(mticker.FixedLocator(major_ticks))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    min_acc = max(0, acc_df["accuracy"].min() - 2) if not acc_df.empty else 0
    max_acc = 100
    if acc_df["accuracy"].max() > 90:
        max_acc = 100
    else:
        max_acc = min(100, acc_df["accuracy"].max() + 2) if not acc_df.empty else 100
    max_time = acc_df["avg_time"].max() + 10 if not acc_df.empty else 40
    ax.set_ylim(min_acc, max_acc)
    # ax.set_ylim(0, 100)
    ax.set_xlim(left=0.0, right=max_time)

    # Add note and finalize
    # plt.figtext(0.5, 0.01, f"Note: Accuracy based on '{accuracy_metric}' assessment.", 
    #             ha='center', fontsize=12, style='italic')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax