# -*- coding: utf-8 -*-
"""Ablacion del umbral de vocabulario `bow_pct_words`: el paper fija este
hiperparametro al 10% para todos los datasets sin mostrar sensibilidad a esta
eleccion, asi que se repite el entrenamiento en varios datasets
representativos, variando bow_pct_words en {5, 10, 20, 50} y manteniendo fijo
el resto de hiperparametros (los ya elegidos por grid-search para cada
dataset, Tabla 8), para aislar el efecto del tamano de vocabulario.

pct=10 no se reentrena: ya existe como el checkpoint actual de BOW2ITM para
cada dataset; sus metricas se leen directamente de ese checkpoint.

Cada pct se entrena en un proceso separado (uso: `python vocab_threshold_ablation.py <dataset> <subset> <pct> dev|final|eval`)
para evitar reutilizar estado de Keras/TensorFlow entre entrenamientos
consecutivos dentro del mismo proceso.

Uso: python vocab_threshold_ablation.py <dataset> <subset> <pct> <dev|final|eval> [gpu]
  dataset/subset: restaurants/gijon, pois/london, amazon/fashion, ...
"""
import json
import os
import sys

dataset_name = sys.argv[1]
subset_name = sys.argv[2]
pct = int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else "train"
gpu = sys.argv[5] if len(sys.argv) > 5 else "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.BOW2ITM import BOW2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

# Rutas/idioma/lr/batch: los mismos que el checkpoint actual (pct=10) de cada
# dataset, para aislar el efecto de bow_pct_words sin reabrir la busqueda de
# hiperparametros. Tomados directamente de sus cfg.json (Tabla 8).
DATASET_DEFAULTS = {
    ("restaurants", "gijon"): {
        "data_path": "/media/nas/datasets/tripadvisor/restaurants/", "language": "es",
        "lr": 0.001, "final_lr": 1e-05, "batch": 1024,
    },
    ("pois", "london"): {
        "data_path": "/media/nas/datasets/tripadvisor/pois/", "language": "es",
        "lr": 0.001, "final_lr": 1e-05, "batch": 1024,
    },
    ("amazon", "fashion"): {
        "data_path": "/media/nas/datasets/amazon/", "language": "en",
        "lr": 0.005, "final_lr": 5e-05, "batch": 8192,
    },
}

defaults = DATASET_DEFAULTS[(dataset_name, subset_name)]

DTS_CFG = {
    "bow_pct_words": pct,
    "data_path": defaults["data_path"],
    "dataset": dataset_name,
    "language": defaults["language"],
    "lemmatization": True,
    "min_df": 5,
    "min_reviews_rst": 100,
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
# NOTA: dataset.DATA["VOCAB_SIZE"] es el vocabulario del tokenizer de SECUENCIAS
# (controlado por n_max_words=-50, fijo, NADA que ver con bow_pct_words). El
# vocabulario BoW real -- el que bow_pct_words controla -- es FEATURES_NAME.
bow_vocab_size = len(dataset.DATA["FEATURES_NAME"])
print(f"{dataset_name}/{subset_name} pct={pct}% -> BOW vocab size = {bow_vocab_size}", flush=True)

mdl_cfg = {"model": MDL_CFG, "session": {"gpu": int(gpu), "mixed_precision": False, "in_md5": False}}
model = BOW2ITM(mdl_cfg, dataset)
print(f"MD5={model.MD5}", flush=True)

if mode == "dev":
    model.train(dev=True, save_model=True)

elif mode == "final":
    model.train(dev=False, save_model=True)

elif mode == "eval":
    # Un proceso nuevo parte de pesos sin entrenar: hay que cargar los pesos finales
    # explicitamente antes de evaluar (train(dev=False, save_model=True) hace
    # load_weights() y retorna de inmediato si el checkpoint final ya existe).
    model.train(dev=False, save_model=True)
    ret = model.evaluate(test=True, user_info=True, verbose=True)
    user_agg = pd.read_csv(f"{model.MODEL_PATH}final_user_agg_eval.csv")
    log_data = pd.read_csv(model.MODEL_PATH + "log.csv")

    row = {
        "dataset": dataset_name, "subset": subset_name,
        "pct": pct,
        "vocab_size": bow_vocab_size,
        "epochs": int(log_data["epoch"].max() + 1),
        "ndcg10_review": float(ret["NDCG@10"].values[0]),
        "recall10_review": float(ret["Recall@10"].values[0]),
        "ndcg10_user": float(user_agg["NDCG@10_user"].mean()),
        "recall10_user": float(user_agg["Recall@10_user"].mean()),
        "md5": model.MD5,
    }
    print(json.dumps(row), flush=True)

    out_path = "results/vocab_ablation_results.csv"
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    rows = [r for r in rows if not (r["dataset"] == dataset_name and r["subset"] == subset_name and r["pct"] == pct)] + [row]
    pd.DataFrame(rows).sort_values(["dataset", "subset", "pct"]).to_csv(out_path, index=False)
    print(f"Guardado en {out_path}", flush=True)

else:
    raise ValueError(f"modo desconocido: {mode}")
