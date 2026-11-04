# -*- coding: utf-8 -*-
"""
Retrains a USEM2ITM/BERT2ITM model on an Amazon dataset FROM SCRATCH, after
fixing the duplicate-reviewId bug in __create_tfdata__ (see
src/models/text_models/USEM2ITM.py and BERT2ITM.py).

Unlike experiments/run_one_user_agg.py, this one IS meant to retrain: the "final"
checkpoints (not the dev/ ones, which are kept intact to avoid repeating the
grid search) were moved beforehand to _pre_reviewid_fix_backup/, so
KerasModelClass.__train_model__ finds no existing checkpoint and genuinely
retrains, using the same number of epochs (derived from the early stopping
already recorded in dev/log.csv) and hyperparameters as the original (buggy)
run.

Usage:
    python legacy/run_one_retrain_amazon_fix.py --model USEM2ITM --dataset amazon --subset fashion --md5 b73849c1...
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['USEM2ITM', 'BERT2ITM'])
parser.add_argument('--dataset', default='amazon')
parser.add_argument('--subset', required=True)
parser.add_argument('--md5', required=True)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

MODEL_CLASSES = {"USEM2ITM": USEM2ITM, "BERT2ITM": BERT2ITM}
MIXED_PRECISION = {"USEM2ITM": True, "BERT2ITM": True}

cfg_path = f"models/{args.model}/{args.dataset}/{args.subset}/{args.md5}/cfg.json"
with open(cfg_path) as f:
    cfg = json.load(f)

dts_cfg = cfg["dataset_config"]
mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}

text_dataset = AmazonDataset(dts_cfg)
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

if model.MD5 != args.md5:
    print(f"FATAL: MD5 mismatch for {args.model}/{args.dataset}/{args.subset}: got {model.MD5}, expected {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if os.path.exists(checkpoint_path):
    print(f"FATAL: a final checkpoint already exists at {model.MODEL_PATH}; it was expected to have been moved to _pre_reviewid_fix_backup/ to force retraining")
    sys.exit(1)

dev_log_path = model.MODEL_PATH + "dev/log.csv"
if not os.path.exists(dev_log_path):
    print(f"FATAL: {dev_log_path} does not exist, cannot determine the number of epochs for the final training run")
    sys.exit(1)

print(f"Retraining {args.model}/{args.dataset}/{args.subset} (md5={args.md5})...")
model.train(dev=False, save_model=True)

print("Evaluating (existing per-review + new per-user metrics)...")
model.evaluate(test=True, user_info=True)

print(f"DONE {args.model}/{args.dataset}/{args.subset}/{args.md5}")
