# -*- coding: utf-8 -*-
"""
Reentrena USEM2ITM/BERT2ITM en Amazon usando la caché post-fix correcta del
dataset (pos_tagging_fix=true, la misma que usa BOW2ITM/ETeR-X y
knn_bow_baseline.py), en vez de una caché pre-fix generada por una ejecucion
independiente del mismo pipeline (con tamaños de TEST ligeramente distintos,
ej. 19960/23435 filas frente a 20696/25160 en la caché correcta -- el
preprocesado de texto tiene algun paso no determinista entre ejecuciones
separadas del mismo pipeline).

A diferencia de run_one_retrain_amazon_fix.py, aqui el dataset_config es NUEVO
por completo (nunca se entreno ningun USEM2ITM/BERT2ITM con esta caché), asi
que no existe una curva de early-stopping (dev/log.csv) que reutilizar: hay
que correr primero la fase dev=True (para obtener el numero de epochs optimo)
y luego dev=False (entrenamiento final), reusando los MISMOS hiperparametros
ya seleccionados en su dia (mismo learning_rate/batch_size/model_version que
el checkpoint antiguo, solo cambia el dataset_config).

Ya incluye el fix de reviewId duplicados (drop_duplicates en __create_tfdata__
de USEM2ITM/BERT2ITM), aplicado permanentemente en el codigo.

Uso:
    python run_one_retrain_amazon_correct_cache.py --model BERT2ITM --subset fashion
"""
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['USEM2ITM', 'BERT2ITM'])
parser.add_argument('--subset', required=True, choices=['fashion', 'digital_music'])
parser.add_argument('--phase', required=True, choices=['dev', 'final'],
                     help="dev y final SIEMPRE en procesos separados: se detecto que entrenar ambas "
                          "fases en el mismo proceso/instancia de modelo deja el learning-rate del "
                          "optimizador sin resetear entre fases para USEM2ITM (linear_decay no tiene "
                          "suelo, acaba en negativo). El pipeline original (FinalTrain.py) siempre "
                          "hizo dev y final en procesos independientes -- aqui se replica ese patron.")
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["KERAS_BACKEND"] = "tensorflow"

from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

MODEL_CLASSES = {"USEM2ITM": USEM2ITM, "BERT2ITM": BERT2ITM}
MIXED_PRECISION = {"USEM2ITM": True, "BERT2ITM": True}

# Hiperparametros ya seleccionados por la busqueda en rejilla original para
# cada combo (ver el cfg.json del checkpoint "final" correspondiente en
# models/{model}/amazon/{subset}/) -- aqui solo se reutilizan, no se repite
# la busqueda.
MODEL_CFGS = {
    ("USEM2ITM", "fashion"): {"model_version": "0", "learning_rate": 1e-05, "final_learning_rate": 1e-07,
                               "epochs": 1000, "batch_size": 2048, "seed": 100,
                               "early_st_first_epoch": 0, "early_st_monitor": "val_loss",
                               "early_st_monitor_mode": "min", "early_st_patience": 50},
    ("USEM2ITM", "digital_music"): {"model_version": "0", "learning_rate": 1e-05, "final_learning_rate": 1e-07,
                                     "epochs": 1000, "batch_size": 2048, "seed": 100,
                                     "early_st_first_epoch": 0, "early_st_monitor": "val_loss",
                                     "early_st_monitor_mode": "min", "early_st_patience": 50},
    ("BERT2ITM", "fashion"): {"model_version": "0", "learning_rate": 0.001, "final_learning_rate": 1e-05,
                               "epochs": 1000, "batch_size": 512, "seed": 100,
                               "early_st_first_epoch": 0, "early_st_monitor": "val_loss",
                               "early_st_monitor_mode": "min", "early_st_patience": 50},
    ("BERT2ITM", "digital_music"): {"model_version": "0", "learning_rate": 1e-05, "final_learning_rate": 1e-07,
                                     "epochs": 1000, "batch_size": 512, "seed": 100,
                                     "early_st_first_epoch": 0, "early_st_monitor": "val_loss",
                                     "early_st_monitor_mode": "min", "early_st_patience": 50},
}

# dts_cfg POST-fix correcto (identico al que usa BOW2ITM), obtenido de
# models/BOW2ITM/amazon/{subset}/*/cfg.json con pos_tagging_fix=true.
BOW_CFG_PATHS = {
    "fashion": "models/BOW2ITM/amazon/fashion/ad184096499d86322cc6c54158ae4fce/cfg.json",
    "digital_music": "models/BOW2ITM/amazon/digital_music/0cca0ace1e36d81d6a5151cfec7ab231/cfg.json",
}

with open(BOW_CFG_PATHS[args.subset]) as f:
    dts_cfg = json.load(f)["dataset_config"]
assert dts_cfg.get("pos_tagging_fix") is True, "dts_cfg no tiene pos_tagging_fix=true, algo va mal"

mdl_cfg_model = MODEL_CFGS[(args.model, args.subset)]
mdl_cfg = {"model": mdl_cfg_model, "session": {"gpu": args.gpu, "mixed_precision": MIXED_PRECISION[args.model], "in_md5": False}}

text_dataset = AmazonDataset(dts_cfg)
model = MODEL_CLASSES[args.model](mdl_cfg, text_dataset)

print(f"MODEL_PATH: {model.MODEL_PATH}")
print(f"TEST size: {len(text_dataset.DATA['TEST'])}")

dev_checkpoint = model.MODEL_PATH + "dev/checkpoint"
final_checkpoint = model.MODEL_PATH + "checkpoint"

if args.phase == "dev":
    if os.path.exists(dev_checkpoint):
        print("Dev ya entrenado para esta config exacta -- nada que hacer.")
        sys.exit(0)
    print("Entrenando dev (early stopping) para obtener el numero de epochs...")
    model.train(dev=True, save_model=True)
    print(f"DEV_DONE {args.model}/amazon/{args.subset} -> {model.MODEL_PATH}")

else:  # final
    if os.path.exists(final_checkpoint):
        print("Ya existe un checkpoint final para esta config exacta -- nada que hacer.")
        sys.exit(0)
    if not os.path.exists(dev_checkpoint):
        print(f"FATAL: no hay dev/checkpoint en {model.MODEL_PATH} -- correr primero --phase dev")
        sys.exit(1)
    print("Entrenamiento final (train+dev completo, proceso nuevo -> optimizador fresco)...")
    model.train(dev=False, save_model=True)

    print("Evaluando (per-review + per-usuario)...")
    model.evaluate(test=True, user_info=True)

    print(f"DONE {args.model}/amazon/{args.subset} -> {model.MODEL_PATH}")
