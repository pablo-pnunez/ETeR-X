# -*- coding: utf-8 -*-
"""
Zero-shot embedding baselines: EmbeddingGemma and Qwen3-Embedding.

Methodology (described in Main.tex, Section~\ref{sec:baselines}): each item is
represented by a profile = mean of the embeddings of a sample of its
training reviews (zero-shot, no fine-tuning). A query (test
review) is embedded with the same model and ranked by cosine similarity against
each item profile. This is a QUERY-based approach (no user profile is built),
consistent with the rest of the paper.

Reports both the per-review metric and the per-user aggregation (mean of
a user's predictions, union of their relevant items), so that the
embedding columns are homogeneous with the rest of tab:results_tk1.

Uses Ollama as the inference server, restricted to a specific GPU
(OLLAMA_HOST/gpu_ids below) so it can run in parallel with
training other models on a different GPU.

Usage: python baselines/embedding_baseline.py [model] [dataset:subset ...]
     model: embgemma | qwen3emb | all (default)
"""
import concurrent.futures
import json
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from codecarbon import EmissionsTracker
from scipy.sparse import csr_matrix  # noqa: F401 (unused, kept for parity with knn_bow_baseline)
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
SEED = 100
MAX_REVIEWS_PER_ITEM = 50  # same as the original ZeroShotRS/ETeR-X (see its comment in dataset_loader.py)

# Port of the Ollama server to use; change it if Ollama is listening on a different
# instance/GPU (useful to avoid competing for GPU with an ongoing training run).
OLLAMA_HOST = os.environ.get("ETERX_OLLAMA_HOST", "http://127.0.0.1:11435")
OLLAMA_GPU_ID = int(os.environ.get("ETERX_OLLAMA_GPU", 0))  # must match the actual GPU used by the instance above
MODELS = {"embgemma": "embeddinggemma", "qwen3emb": "qwen3-embedding:0.6b"}
BATCH_SIZE = 32
N_WORKERS = 4
TIMEOUT_S = 120
MAX_RETRIES = 3


def encode_texts(ollama_name, texts):
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    chunks = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    results = [None] * len(chunks)

    def _work(i_chunk):
        i, chunk = i_chunk
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                r = requests.post(f"{OLLAMA_HOST}/api/embed", json={"model": ollama_name, "input": chunk}, timeout=TIMEOUT_S)
                r.raise_for_status()
                return i, r.json()["embeddings"]
            except requests.exceptions.RequestException as e:
                last_exc = e
                time.sleep(2.0 * (attempt + 1))
        raise last_exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, embs in pool.map(_work, enumerate(chunks)):
            results[i] = embs

    out = np.array([e for chunk in results for e in chunk], dtype=np.float32)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return out / norms


def get_review_text_by_id(all_data, review_ids):
    """Replicates the duplicate-reviewId bug fix: dedup BEFORE indexing."""
    all_data = all_data[["reviewId", "text"]].drop_duplicates(subset="reviewId", keep="first")
    return all_data.set_index("reviewId").loc[review_ids]["text"].fillna("").tolist()


def evaluate_dataset(model_key, dataset, subset, coldstart=False, known=False):
    ollama_name = MODELS[model_key]

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
    n_items = ds.DATA["N_ITEMS"]

    if coldstart:
        # Identical filter to experiments/run_one_coldstart_eval.py / experiments/Main.py's known_users=False
        train_dev_users = train_dev.userId.unique()
        test = test[~test["userId"].isin(train_dev_users)]
        test = test.drop_duplicates(subset=["userId", "id_item"], keep="last")

    if known:
        # Known users only: keep TEST reviews whose author appears in TRAIN_DEV, same
        # population experiments/Baselines.py gives the CF baselines (unlike the default, unfiltered run).
        train_dev_users = train_dev.userId.unique()
        test = test[test["userId"].isin(train_dev_users)]
        test = test.drop_duplicates(subset=["userId", "id_item"], keep="last")

    all_data = pd.read_pickle(ds.DATASET_PATH + "ALL_DATA")

    # --- Bounded per-item sample for the profiles (parity with the original approach) ---
    rng = np.random.RandomState(SEED)
    sampled_idx = (
        train_dev.groupby("id_item", group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), MAX_REVIEWS_PER_ITEM), random_state=SEED))
        .index
    )
    train_sample = train_dev.loc[sampled_idx]

    # The actual computation happens in the Ollama server process, not in this
    # script, so emissions must be measured at the "machine"/GPU level, not at the
    # "process" level (which would see almost nothing, since this script barely does
    # any compute). The GPU id must match the one used by the Ollama instance at OLLAMA_HOST.
    tracker = EmissionsTracker(project_name=f"{model_key}_{dataset}_{subset}", log_level="error",
                                tracking_mode="machine", gpu_ids=[OLLAMA_GPU_ID], measure_power_secs=5,
                                save_to_file=False)
    tracker.start()

    t0 = time.time()
    item_texts = get_review_text_by_id(all_data, train_sample["reviewId"].values)
    item_embs = encode_texts(ollama_name, item_texts)

    item_ids = train_sample["id_item"].values
    dim = item_embs.shape[1]
    item_profiles = np.zeros((n_items, dim), dtype=np.float32)
    counts = np.zeros(n_items, dtype=np.int32)
    for emb, iid in zip(item_embs, item_ids):
        item_profiles[iid] += emb
        counts[iid] += 1
    counts[counts == 0] = 1
    item_profiles = item_profiles / counts[:, None]
    norms = np.linalg.norm(item_profiles, axis=1, keepdims=True)
    norms[norms == 0] = 1
    item_profiles = item_profiles / norms

    # --- Queries: each test review's text, zero-shot ---
    query_texts = get_review_text_by_id(all_data, test["reviewId"].values)
    query_embs = encode_texts(ollama_name, query_texts)
    encode_time_s = time.time() - t0
    emissions_kg = tracker.stop()

    true_items = test["id_item"].values
    user_ids = test["userId"].values
    unique_users, user_row = np.unique(user_ids, return_inverse=True)
    n_users = len(unique_users)
    user_score_sum = np.zeros((n_users, n_items), dtype=np.float32)
    user_review_count = np.zeros(n_users, dtype=np.int32)
    user_relevant = [set() for _ in range(n_users)]

    batch_size = 2048
    ndcgs, hits = [], []
    for start in range(0, len(query_embs), batch_size):
        q_batch = query_embs[start:start + batch_size]
        scores = q_batch @ item_profiles.T  # (batch, n_items), cosine (both L2-normalized)
        true_batch = true_items[start:start + batch_size]
        urow_batch = user_row[start:start + batch_size]

        true_scores = scores[np.arange(len(true_batch)), true_batch]
        ranks = (scores > true_scores[:, None]).sum(axis=1) + 1
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

    user_ndcgs, user_hits = [], []
    for u in range(n_users):
        relevant = user_relevant[u]
        if not relevant:
            continue
        mean_score = user_score_sum[u] / user_review_count[u]
        top_k = np.argsort(-mean_score)[:K]
        user_hits.append(len(set(top_k.tolist()) & relevant) / len(relevant))
        relevant_vec = np.zeros(n_items, dtype=np.int32)
        relevant_vec[list(relevant)] = 1
        user_ndcgs.append(ndcg_score([relevant_vec], [mean_score], k=K))

    return {
        "model": model_key, "dataset": dataset, "subset": subset,
        "n_test": len(test), "n_users": n_users, "encode_time_s": encode_time_s,
        "emissions_g": float(emissions_kg) * 1000 if emissions_kg else None,
        "ndcg10": float(ndcgs.mean()), "recall10": float(hits.mean()),
        "ndcg10_user": float(np.mean(user_ndcgs)), "recall10_user": float(np.mean(user_hits)),
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    coldstart = "--coldstart" in args
    known = "--known" in args
    args = [a for a in args if a not in ("--coldstart", "--known")]
    model_arg = args[0] if args and (args[0] in MODELS or args[0] == "all") else "all"
    combo_args = args[1:] if args and (args[0] in MODELS or args[0] == "all") else args
    targets = [tuple(c.split(":")) for c in combo_args] if combo_args else COMBOS
    model_keys = list(MODELS.keys()) if model_arg == "all" else [model_arg]

    out_path = "results/embedding_results_coldstart.csv" if coldstart else ("results/embedding_results_known.csv" if known else "results/embedding_results.csv")
    rows = []
    if os.path.exists(out_path):
        rows = pd.read_csv(out_path).to_dict("records")

    for model_key in model_keys:
        for dataset, subset in targets:
            if any(r["model"] == model_key and r["dataset"] == dataset and r["subset"] == subset for r in rows):
                print(f"=== {model_key} {dataset}/{subset}: already done, skipping ===", flush=True)
                continue
            print(f"=== {model_key} {dataset}/{subset} ({'coldstart' if coldstart else ('known-only' if known else 'all test users')}) ===", flush=True)
            res = evaluate_dataset(model_key, dataset, subset, coldstart=coldstart, known=known)
            print(f"  n_test={res['n_test']} n_users={res['n_users']} encode_time={res['encode_time_s']:.1f}s | "
                  f"NDCG@10={res['ndcg10']:.4f} Recall@10={res['recall10']:.4f} | "
                  f"NDCG@10_user={res['ndcg10_user']:.4f} Recall@10_user={res['recall10_user']:.4f}", flush=True)
            rows.append(res)
            pd.DataFrame(rows).to_csv(out_path, index=False)  # save incrementally

    print(f"Saved to {out_path}")
