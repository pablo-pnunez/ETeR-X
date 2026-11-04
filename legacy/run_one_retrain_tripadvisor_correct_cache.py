# -*- coding: utf-8 -*-
"""
Retrains USEM2ITM/BERT2ITM on a TripAdvisor dataset (pois/restaurants) using the
post-fix dataset cache (pos_tagging_fix=true) that BOW2ITM/ETeR-X, the CF baselines,
kNN-BoW and the embedding baselines already use, so that every model is evaluated on
the SAME train/test split. The existing USEM2ITM/BERT2ITM TripAdvisor checkpoints were
trained on a pre-fix cache whose TEST set shares only 10-35% of its reviews with the
post-fix one. Same procedure as legacy/run_one_retrain_amazon_correct_cache.py, generalized to
any dataset: hyperparameters (learning rate, batch size, ...) are those of the existing
checkpoint listed in --old-md5 (the grid search is NOT repeated); only dataset_config changes
(copied from the BOW2ITM checkpoint listed in --bow-md5).

dev and final phases ALWAYS run in separate processes (see the Amazon script for why).
After the final phase it evaluates on the full TEST set (per-review + per-user, as usual)
and on the known-users-only TEST (outputs with the _known suffix), and appends the
new md5 to results/tripadvisor_retrain_md5.csv.

Usage:
    python legacy/run_one_retrain_tripadvisor_correct_cache.py --model USEM2ITM --dataset pois --subset paris \
        --old-md5 <existing USEM md5> --bow-md5 <BOW2ITM md5> --phase dev|final
"""
import argparse
import csv
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['USEM2ITM', 'BERT2ITM'])
parser.add_argument('--dataset', required=True, choices=['pois', 'restaurants'])
parser.add_argument('--subset', required=True)
parser.add_argument('--old-md5', required=True, help="md5 of the existing (pre-fix) checkpoint whose hyperparameters are reused")
parser.add_argument('--bow-md5', required=True, help="md5 of the BOW2ITM checkpoint whose (post-fix) dataset_config is used")
parser.add_argument('--phase', required=True, choices=['dev', 'final'])
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fp32', action='store_true', help="train/evaluate in float32 instead of mixed precision (used when the fp16 run diverges to NaN)")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset}
MODEL_CLASSES = {"USEM2ITM": USEM2ITM, "BERT2ITM": BERT2ITM}

with open(f"models/{args.model}/{args.dataset}/{args.subset}/{args.old_md5}/cfg.json") as f:
    mdl_cfg_model = json.load(f)["model"]
with open(f"models/BOW2ITM/{args.dataset}/{args.subset}/{args.bow_md5}/cfg.json") as f:
    dts_cfg = json.load(f)["dataset_config"]
assert dts_cfg.get("pos_tagging_fix") is True, "dts_cfg does not have pos_tagging_fix=true, something is wrong"

mdl_cfg = {"model": mdl_cfg_model, "session": {"gpu": args.gpu, "mixed_precision": not args.fp32, "in_md5": False}}
text_dataset = DATASET_CLASSES[args.dataset](dts_cfg)
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

print(f"MODEL_PATH: {model.MODEL_PATH}")
print(f"TEST size: {len(text_dataset.DATA['TEST'])}")

dev_checkpoint = model.MODEL_PATH + "dev/checkpoint"
final_checkpoint = model.MODEL_PATH + "checkpoint"

if args.phase == "dev":
    if os.path.exists(dev_checkpoint):
        print("Dev already trained for this exact config -- nothing to do.")
        sys.exit(0)
    print("Training dev (early stopping) to obtain the number of epochs...")
    model.train(dev=True, save_model=True)
    print(f"DEV_DONE {args.model}/{args.dataset}/{args.subset} -> {model.MODEL_PATH}")
else:
    if os.path.exists(final_checkpoint):
        print("A final checkpoint already exists for this exact config -- nothing to do.")
        sys.exit(0)
    if not os.path.exists(dev_checkpoint):
        print(f"FATAL: no dev/checkpoint at {model.MODEL_PATH} -- run --phase dev first")
        sys.exit(1)
    print("Final training (full train+dev, new process -> fresh optimizer)...")
    model.train(dev=False, save_model=True)

    print("Evaluating on the full TEST (per-review + per-user)...")
    model.evaluate(test=True, user_info=True)

    print("Evaluating on known users only (per-user, _known suffix)...")
    full_test = text_dataset.DATA["TEST"]
    train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = full_test[full_test["userId"].isin(train_dev_users)].drop_duplicates(subset=["userId", "id_item"], keep="last")
    model.evaluate(test=True, user_info=True, out_tag="_known")
    text_dataset.DATA["TEST"] = full_test

    out = "results/tripadvisor_retrain_md5.csv"
    new_file = not os.path.exists(out)
    with open(out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_file:
            w.writerow(["model", "dataset", "subset", "old_md5", "new_md5"])
        w.writerow([args.model, args.dataset, args.subset, args.old_md5, model.MD5])
    print(f"DONE {args.model}/{args.dataset}/{args.subset} -> {model.MODEL_PATH}")
