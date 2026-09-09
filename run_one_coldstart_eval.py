# -*- coding: utf-8 -*-
"""
Reproduce, para un modelo/dataset ya entrenado, la evaluacion de la segunda
tarea del paper (Section~\ref{sec:results_b}, "user cold-start"): mismos
pesos que la primera tarea/"known users" (no se reentrena nada), pero
filtrando TEST a solo usuarios que NO aparecen en TRAIN_DEV -- replica
exactamente la logica de Main.py cuando known_users=False, que es la unica
implementacion de ese filtro que existe en el repo (Baselines.py, el
generador de la tabla de "known users" para los baselines CF, solo implementa
el filtro contrario: quedarse con usuarios conocidos).

Uso:
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

# --- Filtro cold-start, replica exacta de Main.py known_users=False ---
train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][~text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)
print(f"TEST cold-start (usuarios desconocidos): {len(text_dataset.DATA['TEST'])} reviews")

model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

if model.MD5 != args.md5:
    print(f"FATAL: MD5 no coincide para {args.model}/{args.dataset}/{args.subset}: se obtuvo {model.MD5}, se esperaba {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if not os.path.exists(checkpoint_path):
    print(f"FATAL: no hay checkpoint final en {model.MODEL_PATH}")
    sys.exit(1)

# Carga los pesos ya entrenados (no reentrena)
model.train(dev=False, save_model=True)

# Evalua SOLO sobre el TEST ya filtrado a usuarios desconocidos
ret = model.evaluate(test=True, user_info=False, verbose=False)
print(f"COLDSTART_RESULT {args.model} {args.dataset}/{args.subset} NDCG@10={ret['NDCG@10'].values[0]:.4f} Recall@10={ret['Recall@10'].values[0]:.4f}")
