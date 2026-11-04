# -*- coding: utf-8 -*-
"""
Pareto-front plot: accuracy (NDCG@10 / Recall@10) vs number of parameters.

Companion to pareto_efficiency_plot.py (which uses CO2 emissions as the cost
axis instead) for the Efficiency section (\\subsubsection{Efficiency},
fig:efficiency) of docs/SUSCOM/Final_v1.tex. Same 5 models (ETeR-X, USEM,
BERT + the zero-shot EmbeddingGemma/Qwen3-Embedding baselines, known users).

Each model is reduced to one point: mean number of parameters over the 12
dataset/subset combinations (same convention as the mean NDCG@10/Recall@10
on the y-axis; for ETeR-X/USEM/BERT the parameter count is dataset-dependent,
for the two zero-shot embedding baselines it is the fixed pre-trained model
size) vs the mean NDCG@10/Recall@10. A model is on the Pareto front if no
other model has both fewer parameters and an equal-or-higher metric.

Usage: python pareto_params_plot.py
Output: output/pareto_params_ndcg.pdf, output/pareto_params_recall.pdf
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = "results/"

SET_NAMES = {"Amazon": "AM", "Pois": "POI", "Restaurants": "RST"}
SUBSET_NAMES = {
    "Digital_Music": "MSC", "Fashion": "FSH", "Gijon": "GJN", "Barcelona": "BCN",
    "Madrid": "MDR", "Newyorkcity": "NYC", "Paris": "PRS", "London": "LND",
}

OUR_MODEL = "ETeR-X"
MODELS_TRAINED = {"BOW2ITM": OUR_MODEL, "USEM2ITM": "USEM", "BERT2ITM": "BERT"}
MODELS_EMBEDDING = {"embgemma": "EmbGemma", "qwen3emb": "Qwen3Emb"}
EFFICIENCY_MODELS = [OUR_MODEL, "USEM", "BERT", "EmbGemma", "Qwen3Emb"]
MODEL_COLORS = {
    OUR_MODEL: "#4CAF50", "USEM": "#03A9F4", "BERT": "#ffa726",
    "EmbGemma": "#9C27B0", "Qwen3Emb": "#795548",
}
N_FLIP = 2  # the N highest-x points get a bottom-left label instead of top-right, to stay on the plot


def read_rows(path):
    with open(RESULTS_PATH + path) as f:
        return list(csv.DictReader(f))


def collect_metrics():
    """NDCG@10/Recall@10 (known users, per-user aggregation) per model/dataset/subset."""
    rows = []

    def add(model, dataset, subset, ndcg, recall):
        rows.append({
            "Set": SET_NAMES[dataset.title()], "Subset": SUBSET_NAMES[subset.title()], "Model": model,
            "NDCG@10": float(ndcg), "Recall@10": float(recall),
        })

    for r in read_rows("user_agg_summary_known.csv"):
        if r["model"] == "BOW2ITM" or (r["dataset"] == "amazon" and r["model"] in MODELS_TRAINED):
            add(MODELS_TRAINED[r["model"]], r["dataset"], r["subset"], r["NDCG@10_user"], r["Recall@10_user"])
    for fname, model_name in (("user_agg_summary_known_usem_resplit.csv", "USEM"),
                               ("user_agg_summary_known_bert_resplit.csv", "BERT")):
        for r in read_rows(fname):
            if r["dataset"] != "amazon":
                add(model_name, r["dataset"], r["subset"], r["NDCG@10_user"], r["Recall@10_user"])
    for r in read_rows("embedding_results_known.csv"):
        add(MODELS_EMBEDDING[r["model"]], r["dataset"], r["subset"], r["ndcg10_user"], r["recall10_user"])
    return pd.DataFrame(rows)


def collect_params():
    """Number of parameters per model/dataset/subset, from output/param_results.xlsx."""
    pr = pd.read_excel("output/param_results.xlsx", header=None).iloc[3:15, :].reset_index(drop=True)
    pr.columns = ["set", "subset", "ETeR-X", "USEM", "BERT", "EmbGemma", "Qwen3Emb"]
    pr["set"] = pr["set"].ffill()
    rows = []
    for _, r in pr.iterrows():
        for model in EFFICIENCY_MODELS:
            rows.append({"Set": r["set"], "Subset": r["subset"], "Model": model, "params": float(r[model])})
    return pd.DataFrame(rows)


def pareto_mask(values_sorted_by_x_asc):
    """True for a point whose metric strictly exceeds every point with lower-or-equal x before it."""
    mask, best = [], -np.inf
    for v in values_sorted_by_x_asc:
        keep = v > best
        mask.append(keep)
        if keep:
            best = v
    return mask


def plot(agg, metric, output_file):
    agg = agg.sort_values("params").reset_index(drop=True)
    agg["pareto"] = pareto_mask(agg[metric])
    flip = set(agg["Model"].iloc[-N_FLIP:])  # highest-params points would clip the right edge

    fig, ax = plt.subplots(figsize=(5, 4))
    front = agg[agg["pareto"]]
    ax.plot(front["params"], front[metric], color="lightgrey", linestyle="--", zorder=1, linewidth=1.5)
    for _, row in agg.iterrows():
        color = MODEL_COLORS[row["Model"]]
        marker, size = ("*", 240) if row["pareto"] else ("o", 110)
        ax.scatter(row["params"], row[metric], color=color, marker=marker, s=size,
                   zorder=3, edgecolor="black", linewidth=0.6)
        if row["Model"] in flip:
            xytext, ha, va = (-6, -4), "right", "top"
        else:
            xytext, ha, va = (6, 4), "left", "bottom"
        ax.annotate(row["Model"], (row["params"], row[metric]), textcoords="offset points",
                    xytext=xytext, fontsize=9, ha=ha, va=va)

    ax.set_xscale("log")
    ax.set_xlabel("Mean parameters, 12 datasets (log scale)")
    ax.set_ylabel(f"Mean {metric} (known users)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"output/{output_file}", bbox_inches="tight")
    plt.close(fig)

    print(f"--- {metric} ---")
    print(agg[["Model", "params", metric, "pareto"]].to_string(index=False))


if __name__ == "__main__":
    params = collect_params().groupby("Model", as_index=False)["params"].mean()
    metrics = collect_metrics().groupby("Model", as_index=False)[["NDCG@10", "Recall@10"]].mean()
    agg = params.merge(metrics, on="Model")
    agg = agg[agg["Model"].isin(EFFICIENCY_MODELS)]

    plot(agg, "NDCG@10", "pareto_params_ndcg.pdf")
    plot(agg, "Recall@10", "pareto_params_recall.pdf")
