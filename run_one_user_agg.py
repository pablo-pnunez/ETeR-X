# -*- coding: utf-8 -*-
"""
Helper script to rerun the per-user evaluation (see __per_user_ranking_eval__
in src/models/text_models/RSTModel.py) on an already-trained final model,
without retraining: train(dev=False, save_model=True) detects the existing
checkpoint at MODEL_PATH (derived from the config's md5) and just loads the
weights.

Meant to be invoked once per (model, dataset, subset, md5), as its own
process per combination (see the README for an example loop over every
combination in models/best_models.csv), to avoid reusing the
TensorFlow/GPU session across different models within the same process.

Usage:
    python run_one_user_agg.py --model BOW2ITM --dataset pois --subset paris --md5 8d14e01f...
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['BOW2ITM', 'USEM2ITM', 'BERT2ITM'])
parser.add_argument('--dataset', required=True, choices=['restaurants', 'pois', 'amazon'])
parser.add_argument('--subset', required=True)
parser.add_argument('--md5', required=True, help="md5 of the already-trained final checkpoint under models/<model>/<dataset>/<subset>/<md5>/")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--known', action='store_true', help="restrict TEST to users seen in TRAIN_DEV (same population as the CF baselines in Baselines.py); outputs get a _known suffix")
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
if not os.path.exists(cfg_path):
    print(f"FATAL: {cfg_path} does not exist")
    sys.exit(1)

with open(cfg_path) as f:
    cfg = json.load(f)

dts_cfg = cfg["dataset_config"]
mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}

text_dataset = DATASET_CLASSES[args.dataset](dts_cfg)

if args.known:
    # Same filter as Baselines.py (users in TRAIN_DEV, one row per user/item pair)
    train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep="last")
    print(f"Known-users TEST: {len(text_dataset.DATA['TEST'])} reviews")
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

if model.MD5 != args.md5:
    print(f"FATAL: MD5 mismatch for {args.model}/{args.dataset}/{args.subset}: got {model.MD5}, expected {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if not os.path.exists(checkpoint_path):
    print(f"FATAL: no final checkpoint at {model.MODEL_PATH}, it would need retraining (not done automatically here)")
    sys.exit(1)

# Load the already-trained weights (no retraining, see KerasModelClass.__train_model__ around line 110)
model.train(dev=False, save_model=True)

# Recomputes per-review metrics (already existed) + per-user metrics (new), saves final_user_agg_eval.csv
model.evaluate(test=True, user_info=True, out_tag="_known" if args.known else "")

print(f"DONE {args.model}/{args.dataset}/{args.subset}/{args.md5}")
