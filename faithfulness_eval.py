# -*- coding: utf-8 -*-
"""
Faithfulness of ETeR-X's explanations (intervention test on the trained model).

For every TEST review used as a query, the recommended item is p_hat = top-1.
The explanation is the ranking of the query's terms by W[:, p_hat] (exactly the
weights the paper uses to explain). We then remove the k highest-weighted terms
(re-normalising with L1 and re-scoring with the *full* trained model, including
batch normalisation, so the first-order approximation of Sec. 3.3 is what gets
tested) and measure how far p_hat falls, against:
  - random : k random terms of the same query (mean over --n_random draws)
  - bottom : the k lowest-weighted terms (sanity check, should hurt least)
  - oracle : the k terms whose individual removal hurts p_hat the most
Sufficiency is measured by keeping only the k selected terms instead.

Nothing is retrained: weights are loaded from the checkpoint listed in
models/best_models.csv, like run_one_coldstart_eval.py.

Usage: python faithfulness_eval.py --dataset restaurants --subset gijon [--gpu 0]
                                   [--max_k 5] [--n_random 20] [--max_queries 5000]
Output: results/faithfulness/<dataset>_<subset>.csv (+ _queries.csv)
"""
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, choices=['restaurants', 'pois', 'amazon'])
parser.add_argument('--subset', required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--max_k', type=int, default=5)
parser.add_argument('--n_random', type=int, default=20)
parser.add_argument('--max_queries', type=int, default=5000)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.models.text_models.BOW2ITM import BOW2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

best = pd.read_csv("models/best_models.csv")
row = best[(best.dataset == args.dataset) & (best.subset == args.subset) & (best.model == "BOW2ITM")]
assert len(row) == 1, f"expected 1 BOW2ITM row in best_models.csv, found {len(row)}"
md5 = row.iloc[0]["model_md5"]
with open(f"models/BOW2ITM/{args.dataset}/{args.subset}/{md5}/cfg.json") as f:
    cfg = json.load(f)

text_dataset = DATASET_CLASSES[args.dataset](cfg["dataset_config"])
mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": False, "in_md5": False}}
model = BOW2ITM(mdl_cfg, text_dataset)
assert model.MD5 == md5, f"MD5 mismatch: {model.MD5} != {md5}"
# Load the final checkpoint directly. This is what KerasModelClass.train(save_model=True) does when a checkpoint
# exists, minus building the (huge, dense) training tf.data first, which runs out of GPU memory on the large datasets.
assert os.path.exists(model.MODEL_PATH + "checkpoint"), f"no final checkpoint at {model.MODEL_PATH}"
model.MODEL.load_weights(model.MODEL_PATH + "weights")

# ---- Trained weights -> plain numpy forward pass (dropout off at inference) ----
W, b = model.MODEL.get_layer("bow_2_rst").get_weights()
gamma, beta, mean, var = model.MODEL.get_layer("bow_2_rst_bn").get_weights()
eps = model.MODEL.get_layer("bow_2_rst_bn").epsilon
W, b = W.astype(np.float64), b.astype(np.float64)
scale = gamma / np.sqrt(var + eps)


def logits(X):
    """Pre-sigmoid scores of the full model (dense + inference-mode batch norm)."""
    return ((X @ W + b) - mean) * scale + beta


def l1(X):
    s = X.sum(1, keepdims=True)
    return np.divide(X, s, out=np.zeros_like(X), where=s > 0)


def rank_of(Z, p):
    """1 = top. Computed on logits (monotonic with the sigmoid) to avoid saturation ties."""
    return 1 + (Z > Z[np.arange(len(Z)), p][:, None]).sum(1)


# ---- Queries: every TEST review, per review (no user aggregation) ----
test = text_dataset.DATA["TEST"].reset_index(drop=True)
rng = np.random.default_rng(args.seed)
if len(test) > args.max_queries:
    test = test.iloc[np.sort(rng.choice(len(test), args.max_queries, replace=False))].reset_index(drop=True)
X = np.asarray(text_dataset.DATA["BOW_SEQUENCES"][test.bow.values].todense(), dtype=np.float64)

# Sanity check: our numpy forward pass must reproduce the Keras model
chk = model.MODEL.predict(X[:256].astype(np.float32), verbose=0)
assert np.allclose(1 / (1 + np.exp(-logits(X[:256]))), chk, atol=1e-4), "numpy forward != Keras model"

Z0 = logits(X)
p_hat = Z0.argmax(1)
rank0 = rank_of(Z0, p_hat)  # always 1
prob0 = 1 / (1 + np.exp(-Z0[np.arange(len(X)), p_hat]))
true_item = test.id_item.values
nnz = (X > 0).sum(1)
print(f"{args.dataset}/{args.subset}: {len(X)} queries, top-1 correct {np.mean(p_hat == true_item):.3f}, "
      f"median relevant terms {np.median(nnz):.0f}")

# Weight of every term in the query for the explained item (0 outside the query)
Wq = np.where(X > 0, W[:, p_hat].T, np.nan)  # (n_queries, vocab)


def select(order_key, k, i):
    """Indices of the k terms of query i with the largest `order_key` (NaN = not in query)."""
    return np.argsort(-np.nan_to_num(order_key[i], nan=-np.inf), kind="stable")[:k]


def remove(i, terms):
    x = X[i].copy()
    x[terms] = 0
    return x


def keep_only(i, terms):
    x = np.zeros_like(X[i])
    x[terms] = X[i, terms]
    return x


def measure(xs, ps):
    """rank of p_hat, prob of p_hat and top-1 flip for a batch of modified queries."""
    Z = logits(l1(np.asarray(xs)))
    ps = np.asarray(ps)
    return rank_of(Z, ps), 1 / (1 + np.exp(-Z[np.arange(len(Z)), ps])), Z.argmax(1) != ps


# ---- Oracle: single-term leave-one-out drop of the explained item's logit ----
loo = np.full(X.shape, np.nan)
for i in range(len(X)):
    terms = np.flatnonzero(X[i] > 0)
    if len(terms) == 0:  # no relevant term survives the PoS filter: nothing to explain (never eligible below)
        continue
    Zi = logits(l1(np.stack([remove(i, [t]) for t in terms])))
    loo[i, terms] = Z0[i, p_hat[i]] - Zi[:, p_hat[i]]  # larger = more important

rows, per_query = [], []
for k in range(1, args.max_k + 1):
    elig = np.flatnonzero(nnz > k)  # need at least one term left after removal
    if len(elig) == 0:
        continue
    res = {v: {"rank": [], "prob": [], "flip": [], "suf_rank": []} for v in ["top", "bottom", "random", "oracle"]}
    for i in elig:
        sel = {"top": select(Wq, k, i), "bottom": select(-Wq, k, i), "oracle": select(loo, k, i)}
        terms = np.flatnonzero(X[i] > 0)
        rnd = [rng.choice(terms, k, replace=False) for _ in range(args.n_random)]
        for v, s in sel.items():
            r, p, f = measure([remove(i, s)], [p_hat[i]])
            sr, _, _ = measure([keep_only(i, s)], [p_hat[i]])
            res[v]["rank"].append(r[0]); res[v]["prob"].append(p[0]); res[v]["flip"].append(f[0]); res[v]["suf_rank"].append(sr[0])
        r, p, f = measure([remove(i, s) for s in rnd], [p_hat[i]] * len(rnd))
        sr, _, _ = measure([keep_only(i, s) for s in rnd], [p_hat[i]] * len(rnd))
        res["random"]["rank"].append(r.mean()); res["random"]["prob"].append(p.mean()); res["random"]["flip"].append(f.mean()); res["random"]["suf_rank"].append(sr.mean())

    p0 = prob0[elig]
    for v, d in res.items():
        d = {m: np.asarray(a, dtype=float) for m, a in d.items()}
        row = {"dataset": args.dataset, "subset": args.subset, "k": k, "variant": v, "n_queries": len(elig),
               "frac_eligible": len(elig) / len(X),
               "mean_rank_after": d["rank"].mean(), "mean_rank_drop": (d["rank"] - 1).mean(),
               "median_rank_drop": np.median(d["rank"] - 1),
               "mean_prob_drop": (p0 - d["prob"]).mean(), "flip_rate": d["flip"].mean(),
               "suff_top10": (d["suf_rank"] <= 10).mean(), "suff_mean_rank": d["suf_rank"].mean()}
        row["wilcoxon_p_vs_random"] = np.nan
        if v == "top":  # paired, one-sided: does removing the explanation terms push p_hat further down than random?
            try:
                row["wilcoxon_p_vs_random"] = wilcoxon(d["rank"], np.asarray(res["random"]["rank"], dtype=float), alternative="greater").pvalue
            except ValueError:
                pass
        rows.append(row)
    for j, i in enumerate(elig):
        per_query.append({"k": k, "query_idx": int(i), "id_item_pred": int(p_hat[i]), "id_item_true": int(true_item[i]),
                          **{f"rank_{v}": float(res[v]["rank"][j]) for v in res}})

summary = pd.DataFrame(rows)
os.makedirs("results/faithfulness", exist_ok=True)
tag = f"{args.dataset}_{args.subset}"
summary.to_csv(f"results/faithfulness/{tag}.csv", index=False)
pd.DataFrame(per_query).to_csv(f"results/faithfulness/{tag}_queries.csv", index=False)
pd.set_option("display.width", 200)
print(summary[["k", "variant", "n_queries", "mean_rank_drop", "median_rank_drop", "mean_prob_drop", "flip_rate", "suff_top10", "wilcoxon_p_vs_random"]].round(4).to_string(index=False))
