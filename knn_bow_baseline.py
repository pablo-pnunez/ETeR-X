# -*- coding: utf-8 -*-
"""
kNN-BoW baseline: non-parametric retrieval over Bag-of-Words representations.

Methodology (described in Main.tex, Section~\ref{sec:baselines}): each item is
represented by an aggregated, L1-normalized BoW profile, built from its
TRAIN_DEV reviews (using the same PoS-filtered vocabulary as
ETeR-X). A query (test review) is ranked by cosine similarity against each
item profile. It has no learned parameters, so it requires no
training or hyperparameter search.

Reuses ETeR-X's data pipeline (TextDataset/BOW2ITM): BOW_SEQUENCES is
already L1-normalized per row and positionally aligned with TRAIN_DEV/TEST
via the "bow" column, so a reviewId lookup is never needed. This
makes this baseline immune to the class of duplicate-reviewId bug that
affects USEM2ITM/BERT2ITM on Amazon (see docs/reviews/analisis_revisiones.md
and the comment in BERT2ITM.__create_tfdata__/USEM2ITM.__create_tfdata__).

Usage: python knn_bow_baseline.py [--coldstart]
Output: results/knn_bow_results.csv, or results/knn_bow_results_coldstart.csv with --coldstart
  (dataset, subset, n_test, ndcg10, recall10, ndcg10_user, recall10_user)

--coldstart restricts TEST to reviews from users that do NOT appear in TRAIN_DEV,
the same filter run_one_coldstart_eval.py uses for ETeR-X/USEM/BERT
(identical to Main.py's known_users=False). Since this baseline never
looks at the user's identity, the known-vs-coldstart comparison serves
to empirically confirm that invariance rather than assume it.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score

from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

COMBOS = [
    ("amazon", "fashion"), ("amazon", "digital_music"),
    ("pois", "barcelona"), ("pois", "madrid"), ("pois", "newyorkcity"), ("pois", "paris"), ("pois", "london"),
    ("restaurants", "gijon"), ("restaurants", "barcelona"), ("restaurants", "madrid"),
    ("restaurants", "newyorkcity"), ("restaurants", "paris"),
]

K = 10


def evaluate_dataset(dataset, subset, coldstart=False):
    # The reference checkpoint's MD5 is read from models/best_models.csv instead
    # of locating it with a glob over the checkpoint directory: more than one
    # checkpoint can share the same vocabulary configuration
    # (pos_tagging_fix=true) when there are also ablation checkpoints
    # (bow_pct_words, min_reviews_rst) for the same dataset/subset, so
    # a glob no longer reliably identifies a single candidate.
    best = pd.read_csv("models/best_models.csv")
    row = best[(best.dataset == dataset) & (best.subset == subset) & (best.model == "BOW2ITM")]
    assert len(row) == 1, f"{dataset}/{subset}: expected 1 row in best_models.csv, found {len(row)}"
    md5 = row.iloc[0]["model_md5"]
    with open(f"models/BOW2ITM/{dataset}/{subset}/{md5}/cfg.json") as f:
        cfg = json.load(f)

    ds = DATASET_CLASSES[dataset](cfg["dataset_config"])

    train_dev = ds.DATA["TRAIN_DEV"]
    test = ds.DATA["TEST"]

    if coldstart:
        # Identical filter to run_one_coldstart_eval.py / Main.py's known_users=False
        train_dev_users = train_dev.userId.unique()
        test = test[~test["userId"].isin(train_dev_users)]
        test = test.drop_duplicates(subset=["userId", "id_item"], keep="last")
    bow_seq = ds.DATA["BOW_SEQUENCES"]  # already L1-normalized per row, positionally aligned via "bow"
    n_items = ds.DATA["N_ITEMS"]

    # --- Item profile: sum of the BoW vectors (already L1-norm per review) of
    # its TRAIN_DEV reviews, renormalized L1 ---
    # (group once, instead of filtering train_dev per item -> avoids O(n_items*n_train))
    idx_by_item = train_dev.groupby("id_item")["bow"].apply(lambda s: s.values).to_dict()
    vocab_size = bow_seq.shape[1]
    item_rows = []
    for item_id in range(n_items):
        idx = idx_by_item.get(item_id, np.array([], dtype=int))
        if len(idx) == 0:
            item_rows.append(bow_seq[0:1] * 0)  # item with no reviews in train (shouldn't happen after the min_reviews filter)
            continue
        profile = bow_seq[idx].sum(axis=0)
        item_rows.append(np.asarray(profile).reshape(1, vocab_size))
    item_profiles = csr_matrix(np.vstack(item_rows))
    item_profiles = normalize(item_profiles, axis=1, norm="l1")

    # For cosine similarity with non-negative vectors, L2-normalize and take the dot product
    item_profiles_l2 = normalize(item_profiles, axis=1, norm="l2")

    # --- Query: each test review's own BoW vector (already L1-norm) ---
    query_idx = test["bow"].values
    queries = bow_seq[query_idx]
    queries_l2 = normalize(queries, axis=1, norm="l2")

    true_items = test["id_item"].values

    # --- Prepare per-user accumulators: aggregate by user, not by
    # review, to be homogeneous with Cornac's CF baselines (which evaluate
    # a single prediction per user, not one per interaction) ---
    user_ids = test["userId"].values
    unique_users, user_row = np.unique(user_ids, return_inverse=True)
    n_users = len(unique_users)
    user_score_sum = np.zeros((n_users, n_items), dtype=np.float32)
    user_review_count = np.zeros(n_users, dtype=np.int32)
    user_relevant = [set() for _ in range(n_users)]

    # Batch to avoid materializing the whole dense (n_test x n_items) matrix at once
    batch_size = 4096
    ndcgs, hits = [], []
    for start in range(0, queries_l2.shape[0], batch_size):
        q_batch = queries_l2[start:start + batch_size]
        scores = (q_batch @ item_profiles_l2.T).toarray()  # (batch, n_items)
        true_batch = true_items[start:start + batch_size]
        urow_batch = user_row[start:start + batch_size]

        true_scores = scores[np.arange(len(true_batch)), true_batch]
        ranks = (scores > true_scores[:, None]).sum(axis=1) + 1  # ties favor the true item
        hit = ranks <= K
        ndcg = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)

        ndcgs.append(ndcg)
        hits.append(hit)

        np.add.at(user_score_sum, urow_batch, scores)
        np.add.at(user_review_count, urow_batch, 1)
        for ur, it in zip(urow_batch, true_batch):
            user_relevant[ur].add(int(it))

    ndcgs = np.concatenate(ndcgs)
    hits = np.concatenate(hits)

    # --- Per-user aggregation: mean of the score vectors of their
    # test reviews, union of their relevant items (same criterion as
    # RSTModel.__per_user_ranking_eval__) ---
    user_ndcgs, user_hits = [], []
    for u in range(n_users):
        relevant = user_relevant[u]
        if not relevant:
            continue
        mean_score = user_score_sum[u] / user_review_count[u]
        top_k = np.argsort(-mean_score)[:K]
        n_hit = len(set(top_k.tolist()) & relevant)
        user_hits.append(n_hit / len(relevant))

        relevant_vec = np.zeros(n_items, dtype=np.int32)
        relevant_vec[list(relevant)] = 1
        user_ndcgs.append(ndcg_score([relevant_vec], [mean_score], k=K))

    return {
        "dataset": dataset, "subset": subset, "n_test": len(test), "n_users": n_users,
        "ndcg10": float(ndcgs.mean()), "recall10": float(hits.mean()),
        "ndcg10_user": float(np.mean(user_ndcgs)), "recall10_user": float(np.mean(user_hits)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coldstart", action="store_true")
    args = parser.parse_args()

    out_path = "results/knn_bow_results_coldstart.csv" if args.coldstart else "results/knn_bow_results.csv"

    rows = []
    for dataset, subset in COMBOS:
        print(f"=== {dataset}/{subset} ({'coldstart' if args.coldstart else 'known users'}) ===", flush=True)
        res = evaluate_dataset(dataset, subset, coldstart=args.coldstart)
        print(f"  n_test={res['n_test']} n_users={res['n_users']} "
              f"NDCG@10={res['ndcg10']:.4f} Recall@10={res['recall10']:.4f} | "
              f"NDCG@10_user={res['ndcg10_user']:.4f} Recall@10_user={res['recall10_user']:.4f}", flush=True)
        rows.append(res)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
