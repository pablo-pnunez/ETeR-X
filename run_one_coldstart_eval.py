# -*- coding: utf-8 -*-
"""
Reproduces, for an already-trained model/dataset, the evaluation of the
paper's second task (Section~\ref{sec:results_b}, "user cold-start"): the
same weights as the first task/"known users" (nothing is retrained), but
filtering TEST down to only users that do NOT appear in TRAIN_DEV -- this
replicates exactly the logic Main.py uses when known_users=False, the only
implementation of that filter that exists in the repo (Baselines.py, the
generator of the "known users" table for the CF baselines, only implements
the opposite filter: keep known users).

Usage:
    python run_one_coldstart_eval.py --model BERT2ITM --dataset amazon --subset fashion --md5 7f21293988cf9aa7e9b2b02465f497e3
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['BOW2ITM', 'USEM2ITM', 'BERT2ITM'])
parser.add_argument('--dataset', required=True, choices=['restaurants', 'pois', 'amazon'])
parser.add_argument('--subset', required=True)
parser.add_argument('--md5', required=True)
parser.add_argument('--gpu', type=int, default=0)
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

cfg_path = f"models/{args.model}/{args.dataset}/{args.subset}/{args.md5}/cfg.json"
with open(cfg_path) as f:
    cfg = json.load(f)

dts_cfg = cfg["dataset_config"]
mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}

text_dataset = DATASET_CLASSES[args.dataset](dts_cfg)

# --- Cold-start filter, exact replica of Main.py's known_users=False ---
train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][~text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)
print(f"Cold-start TEST (unknown users): {len(text_dataset.DATA['TEST'])} reviews")

model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

if model.MD5 != args.md5:
    print(f"FATAL: MD5 mismatch for {args.model}/{args.dataset}/{args.subset}: got {model.MD5}, expected {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if not os.path.exists(checkpoint_path):
    print(f"FATAL: no final checkpoint at {model.MODEL_PATH}")
    sys.exit(1)

# Load the already-trained weights (no retraining)
model.train(dev=False, save_model=True)

# Evaluate ONLY on the TEST set already filtered to unknown users
ret = model.evaluate(test=True, user_info=False, verbose=False)
print(f"COLDSTART_RESULT {args.model} {args.dataset}/{args.subset} NDCG@10={ret['NDCG@10'].values[0]:.4f} Recall@10={ret['Recall@10'].values[0]:.4f}")
