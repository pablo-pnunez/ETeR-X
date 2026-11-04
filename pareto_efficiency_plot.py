# -*- coding: utf-8 -*-
"""
Pareto-front plot: accuracy (NDCG@10 / Recall@10) vs CO2 emissions.

For the Efficiency section (\\subsubsection{Efficiency}, fig:efficiency) of
docs/SUSCOM/Final_v1.tex. Reuses the same 5 models already compared there
(ETeR-X, USEM, BERT + the zero-shot EmbeddingGemma/Qwen3-Embedding baselines,
known users), so the numbers are consistent with the emissions already
quoted in the text (summed over the 12 datasets).

Each model is reduced to one point: total CO2 (g, summed over the 12
dataset/subset combinations, same quantity as the "70/90/204/2151/11357 g"
sentence in the paper) vs the mean NDCG@10/Recall@10 (equal weight per
dataset, same convention as the radar plots). A model is on the Pareto
front if no other model has both lower emissions and an equal-or-higher
metric.

Usage: python pareto_efficiency_plot.py
Output: output/pareto_efficiency_ndcg.pdf, output/pareto_efficiency_recall.pdf
"""
import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = "results/"

DATASETS = {
    "restaurants": ["gijon", "barcelona", "madrid", "paris", "newyorkcity"],
    "pois": ["barcelona", "madrid", "paris", "newyorkcity", "london"],
    "amazon": ["fashion", "digital_music"],
}
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
# USEM/BERT sit at the highest-emissions (rightmost) end of the x-axis, so the
# default top-right label offset would run off the plot; place theirs bottom-left instead.
LABEL_OFFSETS = {"USEM": (-6, -4), "BERT": (-6, -4)}
LABEL_ALIGN = {"USEM": ("right", "top"), "BERT": ("right", "top")}


def read_rows(path):
    with open(RESULTS_PATH + path) as f:
        return list(csv.DictReader(f))


def resolve_md5(model, dataset, subset):
    """md5 of the checkpoint behind the paper tables for (model, dataset, subset)."""
    for r in read_rows("tripadvisor_retrain_md5.csv"):
        if (r["model"], r["dataset"], r["subset"]) == (model, dataset, subset):
            return r["new_md5"]
    for r in read_rows("user_agg_summary_known.csv"):
        if (r["model"], r["dataset"], r["subset"]) == (model, dataset, subset):
            return r["md5"]
    raise KeyError((model, dataset, subset))


def collect_co2():
    """Total train/encode emissions (g of CO2) per model/dataset/subset."""
    emb = {(r["model"], r["dataset"], r["subset"]): r for r in read_rows("embedding_results.csv")}
    rows = []
    for dataset, subsets in DATASETS.items():
        for subset in subsets:
            base = dict(Set=SET_NAMES[dataset.title()], Subset=SUBSET_NAMES[subset.title()])
            for model_key, name in MODELS_TRAINED.items():
                md5 = resolve_md5(model_key, dataset, subset)
                path = f"models/{model_key}/{dataset}/{subset}/{md5}/"
                # emissions_train.csv stores the CUMULATIVE emissions (kg); the last row is the total
                co2_g = 1000 * pd.read_csv(path + "emissions_train.csv")["emissions"].iloc[-1]
                rows.append({**base, "Model": name, "co2": co2_g})
            for key, name in MODELS_EMBEDDING.items():
                e = emb[(key, dataset, subset)]
                rows.append({**base, "Model": name, "co2": float(e["emissions_g"])})
    return pd.DataFrame(rows)


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
    agg = agg.sort_values("co2").reset_index(drop=True)
    agg["pareto"] = pareto_mask(agg[metric])

    fig, ax = plt.subplots(figsize=(5, 4))
    front = agg[agg["pareto"]]
    ax.plot(front["co2"], front[metric], color="lightgrey", linestyle="--", zorder=1, linewidth=1.5)
    for _, row in agg.iterrows():
        color = MODEL_COLORS[row["Model"]]
        marker, size = ("*", 240) if row["pareto"] else ("o", 110)
        ax.scatter(row["co2"], row[metric], color=color, marker=marker, s=size,
                   zorder=3, edgecolor="black", linewidth=0.6)
        ha, va = LABEL_ALIGN.get(row["Model"], ("left", "bottom"))
        ax.annotate(row["Model"], (row["co2"], row[metric]), textcoords="offset points",
                    xytext=LABEL_OFFSETS.get(row["Model"], (6, 4)), fontsize=9, ha=ha, va=va)

    ax.set_xscale("log")
    ax.set_xlabel(r"Total emissions, 12 datasets (g CO$_2$, log scale)")
    ax.set_ylabel(f"Mean {metric} (known users)")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(f"output/{output_file}", bbox_inches="tight")
    plt.close(fig)

    print(f"--- {metric} ---")
    print(agg[["Model", "co2", metric, "pareto"]].to_string(index=False))


if __name__ == "__main__":
    co2 = collect_co2().groupby("Model", as_index=False)["co2"].sum()
    metrics = collect_metrics().groupby("Model", as_index=False)[["NDCG@10", "Recall@10"]].mean()
    agg = co2.merge(metrics, on="Model")
    agg = agg[agg["Model"].isin(EFFICIENCY_MODELS)]

    plot(agg, "NDCG@10", "pareto_efficiency_ndcg.pdf")
    plot(agg, "Recall@10", "pareto_efficiency_recall.pdf")
