# -*- coding: utf-8 -*-
"""
Reentrena, DESDE CERO, un modelo USEM2ITM/BERT2ITM en un dataset de Amazon,
tras el fix del bug de reviewId duplicados en __create_tfdata__ (ver
src/models/text_models/USEM2ITM.py y BERT2ITM.py).

A diferencia de run_one_user_agg.py, aqui SI se espera que se reentrene: los
checkpoints "finales" (no los de dev/, que se conservan intactos para no
repetir el grid-search) se movieron previamente a _pre_reviewid_fix_backup/,
asi que KerasModelClass.__train_model__ no encuentra un checkpoint existente
y entrena de verdad, usando el mismo numero de epochs (derivado del
early-stopping ya registrado en dev/log.csv) e hiperparametros que la
ejecucion original (buggy).

Uso:
    python run_one_retrain_amazon_fix.py --model USEM2ITM --dataset amazon --subset fashion --md5 b73849c1...
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
    print(f"FATAL: MD5 no coincide para {args.model}/{args.dataset}/{args.subset}: se obtuvo {model.MD5}, se esperaba {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if os.path.exists(checkpoint_path):
    print(f"FATAL: ya existe un checkpoint final en {model.MODEL_PATH}; se esperaba que estuviera movido a _pre_reviewid_fix_backup/ para forzar el reentrenamiento")
    sys.exit(1)

dev_log_path = model.MODEL_PATH + "dev/log.csv"
if not os.path.exists(dev_log_path):
    print(f"FATAL: no existe {dev_log_path}, no se puede determinar el numero de epochs del entrenamiento final")
    sys.exit(1)

print(f"Reentrenando {args.model}/{args.dataset}/{args.subset} (md5={args.md5})...")
model.train(dev=False, save_model=True)

print("Evaluando (per-review existente + per-usuario nuevo)...")
model.evaluate(test=True, user_info=True)

print(f"DONE {args.model}/{args.dataset}/{args.subset}/{args.md5}")
