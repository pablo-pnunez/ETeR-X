# -*- coding: utf-8 -*-
"""
Per-review NDCG@10/Recall@10 of an already-trained model on two disjoint TEST
partitions, with nothing retrained: reviews by users seen in TRAIN_DEV
("known-only", the same population experiments/Baselines.py gives the CF baselines) and
reviews by users absent from TRAIN_DEV ("cold", experiments/Main.py's known_users=False).
Both partitions keep one row per (user, item) pair, like experiments/Baselines.py.

Appends one row per partition to results/review_eval_known_vs_cold.csv.

Usage:
    python experiments/run_one_known_vs_cold_eval.py --model BOW2ITM --dataset pois --subset paris --md5 <md5>
"""
import argparse
import csv
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['BOW2ITM', 'USEM2ITM', 'BERT2ITM'])
parser.add_argument('--dataset', required=True, choices=['restaurants', 'pois', 'amazon'])
parser.add_argument('--subset', required=True)
parser.add_argument('--md5', required=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--out', default='results/review_eval_known_vs_cold.csv')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}
MODEL_CLASSES = {"BOW2ITM": BOW2ITM, "USEM2ITM": USEM2ITM, "BERT2ITM": BERT2ITM}
MIXED_PRECISION = {"BOW2ITM": False, "USEM2ITM": True, "BERT2ITM": True}

with open(f"models/{args.model}/{args.dataset}/{args.subset}/{args.md5}/cfg.json") as f:
    cfg = json.load(f)

mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}
text_dataset = DATASET_CLASSES[args.dataset](cfg["dataset_config"])
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)
if model.MD5 != args.md5:
    print(f"FATAL: MD5 mismatch: got {model.MD5}, expected {args.md5}")
    sys.exit(1)
if not os.path.exists(model.MODEL_PATH + "checkpoint"):
    print(f"FATAL: no final checkpoint at {model.MODEL_PATH}")
    sys.exit(1)
model.train(dev=False, save_model=True)  # loads the existing weights

original_test = text_dataset.DATA["TEST"]
train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
is_known = original_test["userId"].isin(train_dev_users)

out = args.out
new_file = not os.path.exists(out)
with open(out, "a", newline="") as fh:
    w = csv.writer(fh)
    if new_file:
        w.writerow(["model", "dataset", "subset", "md5", "partition", "n_test", "ndcg10", "recall10"])
    for name, part in (("known", original_test[is_known]), ("cold", original_test[~is_known])):
        text_dataset.DATA["TEST"] = part.drop_duplicates(subset=["userId", "id_item"], keep="last")
        ret = model.evaluate(test=True, user_info=False, verbose=False)
        row = [args.model, args.dataset, args.subset, args.md5, name, len(text_dataset.DATA["TEST"]),
               float(ret["NDCG@10"].values[0]), float(ret["Recall@10"].values[0])]
        w.writerow(row)
        print("RESULT", row, flush=True)
