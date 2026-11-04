# -*- coding: utf-8 -*-
"""Ablation of the minimum-reviews-per-item threshold (`min_reviews_rst`): the
paper uses 100 without showing sensitivity to this choice (the usual standard
in RS is 5-20 interactions). Repeated on the same representative datasets as
the vocabulary ablation (RST Gijon, POI London, Amazon Fashion), varying
min_reviews_rst over {5, 20, 50, 100} while keeping bow_pct_words=10 and the
rest of the hyperparameters fixed (already chosen by grid search for each
dataset, Table 8).

Unlike the vocabulary ablation, changing min_reviews_rst changes the item
universe itself (items with fewer reviews than the threshold are removed
entirely, along with all their reviews): a lower threshold keeps more items
(a potentially harder task, more classes) and more data. So, besides
NDCG@10/Recall@10, N_ITEMS and the TEST set size are recorded at each
threshold, to contextualize any change in the metrics (it isn't necessarily
"the model got better/worse", it could be "the task changed").

min_reviews_rst=100 is not retrained: it already exists as the current
BOW2ITM checkpoint for each dataset; its metrics are read directly from that checkpoint.

Each threshold is trained in a separate process, same pattern as
experiments/vocab_threshold_ablation.py.

Usage: python experiments/min_reviews_ablation.py <dataset> <subset> <min_reviews> <dev|final|eval> [gpu]
  dataset/subset: restaurants/gijon, pois/london, amazon/fashion, ...
"""
import json
import os
import sys

dataset_name = sys.argv[1]
subset_name = sys.argv[2]
min_reviews = int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "train"
gpu = sys.argv[5] if len(sys.argv) > 5 else "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
from src.config import data_path
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.BOW2ITM import BOW2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

# Same lr/batch as the current checkpoint (min_reviews_rst=100) for each
# dataset (Table 8), to isolate the effect of the threshold without
# reopening the hyperparameter search. Same 3 datasets as the bow_pct_words ablation.
DATASET_DEFAULTS = {
    ("restaurants", "gijon"): {
        "data_path": data_path("tripadvisor", "restaurants"), "language": "es",
        "lr": 0.001, "final_lr": 1e-05, "batch": 1024,
    },
    ("pois", "london"): {
        "data_path": data_path("tripadvisor", "pois"), "language": "es",
        "lr": 0.001, "final_lr": 1e-05, "batch": 1024,
    },
    ("amazon", "fashion"): {
        "data_path": data_path("amazon"), "language": "en",
        "lr": 0.005, "final_lr": 5e-05, "batch": 8192,
    },
}

defaults = DATASET_DEFAULTS[(dataset_name, subset_name)]

DTS_CFG = {
    "bow_pct_words": 10,
    "data_path": defaults["data_path"],
    "dataset": dataset_name,
    "language": defaults["language"],
    "lemmatization": True,
    "min_df": 5,
    "min_reviews_rst": min_reviews,
    "min_reviews_usr": 1,
    "n_max_words": -50,
    "pos_tagging_fix": True,
    "presencia": False,
    "remove_accents": True,
    "remove_numbers": True,
    "remove_stopwords": 2,
    "save_path": "data/",
    "seed": 100,
    "subset": subset_name,
    "test_dev_split": 0.1,
    "text_column": "text",
    "truncate_padding": True,
}

MDL_CFG = {
    "model_version": "0",
    "learning_rate": defaults["lr"],
    "final_learning_rate": defaults["final_lr"],
    "epochs": 1000,
    "batch_size": defaults["batch"],
    "seed": 100,
    "early_st_first_epoch": 0,
    "early_st_monitor": "val_loss",
    "early_st_monitor_mode": "min",
    "early_st_patience": 50,
}

dataset = DATASET_CLASSES[dataset_name](DTS_CFG)
n_items = int(dataset.DATA["N_ITEMS"])
n_test = len(dataset.DATA["TEST"])
n_train_dev = len(dataset.DATA["TRAIN_DEV"])
print(f"{dataset_name}/{subset_name} min_reviews={min_reviews} -> N_ITEMS={n_items} "
      f"N_TRAIN_DEV={n_train_dev} N_TEST={n_test}", flush=True)

mdl_cfg = {"model": MDL_CFG, "session": {"gpu": int(gpu), "mixed_precision": False, "in_md5": False}}
model = BOW2ITM(mdl_cfg, dataset)
print(f"MD5={model.MD5}", flush=True)

if mode == "dev":
    model.train(dev=True, save_model=True)

elif mode == "final":
    model.train(dev=False, save_model=True)

elif mode == "eval":
    # A new process starts from untrained weights: the final weights must be
    # loaded explicitly before evaluating.
    model.train(dev=False, save_model=True)
    # evaluate()/predict() use CONFIG["model"]["batch_size"] (the TRAINING one,
    # e.g. 8192 for Amazon) for internal batching. With low min_reviews
    # thresholds the number of items (n_items) can grow well beyond the usual
    # range (Amazon Fashion: 731 -> 27694 items at min_reviews=5), and some
    # full-catalog metrics (Recall@50, NDCG@100, etc.) build one-hot tensors
    # of size batch x n_items (or larger), causing GPU OOM with the training
    # batch size. Reducing the batch ONLY for evaluation is safe: it doesn't
    # affect the already-trained weights or the paths (MODEL_PATH was already
    # fixed in __init__), only the size of the intermediate tensors in evaluate/predict.
    if n_items > 5000:
        # The worst case observed (Recall@50 with broadcast_weights) uses a tensor
        # ~ batch x 50 x n_items x 4 bytes; several full-catalog metrics can
        # coexist without being freed within the same evaluate() call, so a
        # generous margin is left (~50MB budget per tensor of that kind)
        # rather than computing the "exact" batch size.
        eval_batch = max(8, int(50e6 / (50 * n_items * 4)))
        print(f"n_items={n_items} is large; reducing the evaluation batch from "
              f"{model.CONFIG['model']['batch_size']} to {eval_batch} to avoid OOM", flush=True)
        model.CONFIG["model"]["batch_size"] = eval_batch
    ret = model.evaluate(test=True, user_info=True, verbose=True)
    user_agg = pd.read_csv(f"{model.MODEL_PATH}final_user_agg_eval.csv")
    log_data = pd.read_csv(model.MODEL_PATH + "log.csv")

    row = {
        "dataset": dataset_name, "subset": subset_name,
        "min_reviews": min_reviews,
        "n_items": n_items,
        "n_train_dev": n_train_dev,
        "n_test": n_test,
        "epochs": int(log_data["epoch"].max() + 1),
        "ndcg10_review": float(ret["NDCG@10"].values[0]),
        "recall10_review": float(ret["Recall@10"].values[0]),
        "ndcg10_user": float(user_agg["NDCG@10_user"].mean()),
        "recall10_user": float(user_agg["Recall@10_user"].mean()),
        "md5": model.MD5,
    }
    print(json.dumps(row), flush=True)

    out_path = "results/min_reviews_ablation_results.csv"
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    rows = [r for r in rows if not (r["dataset"] == dataset_name and r["subset"] == subset_name and r["min_reviews"] == min_reviews)] + [row]
    pd.DataFrame(rows).sort_values(["dataset", "subset", "min_reviews"]).to_csv(out_path, index=False)
    print(f"Saved to {out_path}", flush=True)

else:
    raise ValueError(f"unknown mode: {mode}")
