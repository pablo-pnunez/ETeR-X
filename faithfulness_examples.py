# -*- coding: utf-8 -*-
"""
Real (non-toy) explanation examples for the faithfulness section of the paper.

For one dataset per domain, picks test queries with a FIXED rule (no cherry-picking):
among the test reviews whose top-1 recommendation is the item the review is actually
about and that have between --min_terms and --max_terms relevant terms, take the first
--n_examples after a seeded shuffle. For each one it prints the query terms ranked by
W[:, p_hat] (the explanation), the top-3 recommended items, and the effect on p_hat of
removing the top-1 / top-2 explanation terms vs. random terms (same intervention as
faithfulness_eval.py: L1 re-normalisation, full model incl. batch norm).

The TEST dataframe stores the already-processed relevant terms, not the raw review text,
so the query is shown as its relevant terms (what the model actually sees).

Usage: python faithfulness_examples.py [--gpu 1] [--n_examples 3] [--seed 0]
Output: results/faithfulness/examples.csv (+ printed report)
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--n_examples', type=int, default=3)
parser.add_argument('--min_terms', type=int, default=5)
parser.add_argument('--max_terms', type=int, default=8)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.models.text_models.BOW2ITM import BOW2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}
COMBOS = [("restaurants", "gijon"), ("pois", "london"), ("amazon", "fashion")]
best = pd.read_csv("models/best_models.csv")
out_rows = []

for dataset, subset in COMBOS:
    row = best[(best.dataset == dataset) & (best.subset == subset) & (best.model == "BOW2ITM")]
    md5 = row.iloc[0]["model_md5"]
    with open(f"models/BOW2ITM/{dataset}/{subset}/{md5}/cfg.json") as f:
        cfg = json.load(f)
    text_dataset = DATASET_CLASSES[dataset](cfg["dataset_config"])
    model = BOW2ITM({"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": False, "in_md5": False}}, text_dataset)
    assert model.MD5 == md5
    assert os.path.exists(model.MODEL_PATH + "checkpoint")
    model.MODEL.load_weights(model.MODEL_PATH + "weights")

    W, b = model.MODEL.get_layer("bow_2_rst").get_weights()
    gamma, beta, mean, var = model.MODEL.get_layer("bow_2_rst_bn").get_weights()
    eps = model.MODEL.get_layer("bow_2_rst_bn").epsilon
    W, b = W.astype(np.float64), b.astype(np.float64)
    scale = gamma / np.sqrt(var + eps)
    logits = lambda X: ((X @ W + b) - mean) * scale + beta
    l1 = lambda X: X / np.maximum(X.sum(-1, keepdims=True), 1e-12)

    test = text_dataset.DATA["TEST"].reset_index(drop=True)
    names = test.drop_duplicates("id_item").set_index("id_item")["name"].to_dict()
    feats = np.asarray(text_dataset.DATA["FEATURES_NAME"])
    X = np.asarray(text_dataset.DATA["BOW_SEQUENCES"][test.bow.values].todense(), dtype=np.float64)
    Z0 = logits(X)
    p_hat = Z0.argmax(1)
    nnz = (X > 0).sum(1)
    ok = np.flatnonzero((p_hat == test.id_item.values) & (nnz >= args.min_terms) & (nnz <= args.max_terms))
    rng = np.random.default_rng(args.seed)
    picked = rng.permutation(ok)[:args.n_examples]
    print(f"\n##### {dataset}/{subset}: {len(ok)} queries meet the rule; showing {len(picked)}")

    def rank_after(i, removed):
        x = X[i].copy(); x[removed] = 0
        z = logits(l1(x[None]))[0]
        return int(1 + (z > z[p_hat[i]]).sum())

    for i in picked:
        p = p_hat[i]
        terms = np.flatnonzero(X[i] > 0)
        order = terms[np.argsort(-W[terms, p], kind="stable")]
        top3 = np.argsort(-Z0[i])[:3]
        rnd = [rng.choice(terms, 2, replace=False) for _ in range(20)]
        rec = {"dataset": dataset, "subset": subset, "query_idx": int(i), "recommended": names.get(p, p),
               "top3": " | ".join(str(names.get(t, t)) for t in top3),
               "terms_by_weight": ", ".join(f"{feats[t]} ({W[t, p]:.2f})" for t in order),
               "rank_remove_top1": rank_after(i, order[:1]), "rank_remove_top2": rank_after(i, order[:2]),
               "rank_remove_rand1_mean": float(np.mean([rank_after(i, [r[0]]) for r in rnd])),
               "rank_remove_rand2_mean": float(np.mean([rank_after(i, r) for r in rnd])),
               "rank_remove_bottom1": rank_after(i, order[-1:])}
        out_rows.append(rec)
        print(f"\nquery #{i}: relevant terms = {', '.join(feats[terms])}")
        for k, v in rec.items():
            if k not in ("dataset", "subset", "query_idx"):
                print(f"   {k}: {v}")

os.makedirs("results/faithfulness", exist_ok=True)
pd.DataFrame(out_rows).to_csv("results/faithfulness/examples.csv", index=False)
