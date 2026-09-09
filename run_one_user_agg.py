# -*- coding: utf-8 -*-
"""
Script auxiliar para relanzar la evaluacion per-usuario (ver
__per_user_ranking_eval__ en src/models/text_models/RSTModel.py) sobre un
modelo final ya entrenado, sin reentrenar: train(dev=False, save_model=True)
detecta el checkpoint existente en el MODEL_PATH (derivado del md5 de la
config) y solo carga los pesos.

Pensado para invocarse una vez por (modelo, dataset, subset, md5), en un
proceso independiente por combinacion (ver README para un ejemplo de bucle
sobre todas las combinaciones a partir de models/best_models.csv), para
evitar reutilizar la sesion de TensorFlow/GPU entre distintos modelos dentro
del mismo proceso.

Uso:
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
parser.add_argument('--md5', required=True, help="md5 de la carpeta final ya entrenada en models/<model>/<dataset>/<subset>/<md5>/")
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
if not os.path.exists(cfg_path):
    print(f"FATAL: no existe {cfg_path}")
    sys.exit(1)

with open(cfg_path) as f:
    cfg = json.load(f)

dts_cfg = cfg["dataset_config"]
mdl_cfg = {"model": cfg["model"], "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}

text_dataset = DATASET_CLASSES[args.dataset](dts_cfg)
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

if model.MD5 != args.md5:
    print(f"FATAL: MD5 no coincide para {args.model}/{args.dataset}/{args.subset}: se obtuvo {model.MD5}, se esperaba {args.md5}")
    sys.exit(1)

checkpoint_path = model.MODEL_PATH + "checkpoint"
if not os.path.exists(checkpoint_path):
    print(f"FATAL: no hay checkpoint final en {model.MODEL_PATH}, habria que reentrenar (no se hace automaticamente aqui)")
    sys.exit(1)

# Carga los pesos ya entrenados (no reentrena, ver KerasModelClass.__train_model__ linea ~110)
model.train(dev=False, save_model=True)

# Recalcula metricas per-review (ya existian) + per-usuario (nuevo), guarda final_user_agg_eval.csv
model.evaluate(test=True, user_info=True)

print(f"DONE {args.model}/{args.dataset}/{args.subset}/{args.md5}")
