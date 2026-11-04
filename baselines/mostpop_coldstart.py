# -*- coding: utf-8 -*-
"""
Real MostPop numbers under the cold-start protocol (unknown users in TEST),
to complete the parity of the 7 user-identity-agnostic baselines mentioned in
Appendix B / Section~\ref{sec:results_b} of Main.tex.

MOSTPOP2ITM has no weights to load (its "model" is a fixed popularity vector,
computed directly from TRAIN_DEV in get_model(), see
src/models/text_models/MOSTPOP2ITM.py): it is enough to build it on top of
the already-loaded dataset, no train()/checkpoint needed, the same way
experiments/run_one_coldstart_eval.py does for BOW2ITM/USEM2ITM/BERT2ITM (but that script
does not support MOSTPOP2ITM). Reports, for each dataset, both NDCG@10/Recall@10
on the full TEST set ("known", unfiltered -- same definition used by
baselines/knn_bow_baseline.py/embedding_baseline.py without --coldstart) and on the
TEST set filtered to users absent from TRAIN_DEV ("coldstart", identical
filter to experiments/Main.py's known_users=False / experiments/run_one_coldstart_eval.py).

Usage: python baselines/mostpop_coldstart.py [gpu]
Output: results/mostpop_coldstart_results.csv
"""
import json
import os
import sys

gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.MOSTPOP2ITM import MOSTPOP2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

COMBOS = [
    ("amazon", "fashion"), ("amazon", "digital_music"),
    ("pois", "barcelona"), ("pois", "madrid"), ("pois", "newyorkcity"), ("pois", "paris"), ("pois", "london"),
    ("restaurants", "gijon"), ("restaurants", "barcelona"), ("restaurants", "madrid"),
    ("restaurants", "newyorkcity"), ("restaurants", "paris"),
]

best = pd.read_csv("models/best_models.csv")

rows = []
for dataset, subset in COMBOS:
    row = best[(best.dataset == dataset) & (best.subset == subset) & (best.model == "MOSTPOP2ITM")]
    assert len(row) == 1, f"{dataset}/{subset}: expected 1 row in best_models.csv, found {len(row)}"
    md5 = row.iloc[0]["model_md5"]
    with open(f"models/MOSTPOP2ITM/{dataset}/{subset}/{md5}/cfg.json") as f:
        cfg = json.load(f)

    text_dataset = DATASET_CLASSES[dataset](cfg["dataset_config"])
    n_test_known = len(text_dataset.DATA["TEST"])

    mdl_cfg = {"model": cfg["model"], "session": {"gpu": int(gpu), "mixed_precision": False, "in_md5": False}}
    model = MOSTPOP2ITM(mdl_cfg, text_dataset)
    assert model.MD5 == md5, f"MD5 mismatch for {dataset}/{subset}: {model.MD5} != {md5}"
    model.train(dev=False, save_model=True)

    # --- "known": full TEST set, unfiltered (same definition as baselines/knn_bow_baseline.py/embedding_baseline.py without --coldstart) ---
    ret_known = model.evaluate(test=True, user_info=False, verbose=False)

    # --- "known_only": TEST restricted to users seen in TRAIN_DEV (same population as the CF baselines) ---
    original_test = text_dataset.DATA["TEST"]
    known_train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = original_test[original_test["userId"].isin(known_train_dev_users)].drop_duplicates(subset=["userId", "id_item"], keep="last")
    n_test_knownonly = len(text_dataset.DATA["TEST"])
    ret_knownonly = model.evaluate(test=True, user_info=False, verbose=False)
    text_dataset.DATA["TEST"] = original_test

    # --- "coldstart": identical filter to experiments/Main.py's known_users=False / experiments/run_one_coldstart_eval.py ---
    train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][~text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep="last")
    n_test_cold = len(text_dataset.DATA["TEST"])
    ret_cold = model.evaluate(test=True, user_info=False, verbose=False)

    row_out = {
        "dataset": dataset, "subset": subset,
        "n_test_known": n_test_known, "n_test_cold": n_test_cold, "n_test_knownonly": n_test_knownonly,
        "ndcg10_knownonly": float(ret_knownonly["NDCG@10"].values[0]), "recall10_knownonly": float(ret_knownonly["Recall@10"].values[0]),
        "ndcg10_known": float(ret_known["NDCG@10"].values[0]),
        "recall10_known": float(ret_known["Recall@10"].values[0]),
        "ndcg10_cold": float(ret_cold["NDCG@10"].values[0]),
        "recall10_cold": float(ret_cold["Recall@10"].values[0]),
    }
    print(json.dumps(row_out), flush=True)
    rows.append(row_out)

df = pd.DataFrame(rows)
df.to_csv("results/mostpop_coldstart_results.csv", index=False)
print("\n=== Averages (12 datasets) ===")
print(df[["ndcg10_known", "recall10_known", "ndcg10_cold", "recall10_cold"]].mean())
print("Saved to results/mostpop_coldstart_results.csv")
