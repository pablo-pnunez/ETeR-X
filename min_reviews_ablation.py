# -*- coding: utf-8 -*-
"""Ablacion del umbral minimo de reseñas por item (`min_reviews_rst`): el paper
usa 100 sin mostrar sensibilidad a esta eleccion (el estandar habitual en RS
es 5-20 interacciones). Se repite en los mismos datasets representativos que
la ablacion de vocabulario (RST Gijon, POI London, Amazon Fashion), variando
min_reviews_rst en {5, 20, 50, 100} y manteniendo fijos bow_pct_words=10 y el
resto de hiperparametros (los ya elegidos por grid-search para cada dataset,
Tabla 8).

A diferencia de la ablacion de vocabulario, cambiar min_reviews_rst cambia el
universo de items en si (items con menos reseñas que el umbral se eliminan
por completo, junto con todas sus reseñas): un umbral mas bajo retiene mas
items (tarea potencialmente mas dificil, mas clases) y mas datos. Por tanto,
ademas de NDCG@10/Recall@10 se registra N_ITEMS y el tamaño del TEST set en
cada umbral, para poder contextualizar cualquier cambio en las metricas (no
es directamente "el modelo mejora/empeora", puede ser "la tarea cambio").

min_reviews_rst=100 no se reentrena: ya existe como el checkpoint actual de
BOW2ITM para cada dataset; sus metricas se leen directamente de ese checkpoint.

Cada umbral se entrena en un proceso separado, mismo patron que
vocab_threshold_ablation.py.

Uso: python min_reviews_ablation.py <dataset> <subset> <min_reviews> <dev|final|eval> [gpu]
  dataset/subset: restaurants/gijon, pois/london, amazon/fashion, ...
"""
import json
import os
import sys

dataset_name = sys.argv[1]
subset_name = sys.argv[2]
min_reviews = int(sys.argv[3])
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

# Mismos lr/batch que el checkpoint actual (min_reviews_rst=100) de cada
# dataset (Tabla 8), para aislar el efecto del umbral sin reabrir la busqueda
# de hiperparametros. Mismos 3 datasets que la ablacion de bow_pct_words.
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
    "bow_pct_words": 10,
    "data_path": defaults["data_path"],
    "dataset": dataset_name,
    "language": defaults["language"],
    "lemmatization": True,
    "min_df": 5,
    "min_reviews_rst": min_reviews,
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
n_items = int(dataset.DATA["N_ITEMS"])
n_test = len(dataset.DATA["TEST"])
n_train_dev = len(dataset.DATA["TRAIN_DEV"])
print(f"{dataset_name}/{subset_name} min_reviews={min_reviews} -> N_ITEMS={n_items} "
      f"N_TRAIN_DEV={n_train_dev} N_TEST={n_test}", flush=True)

mdl_cfg = {"model": MDL_CFG, "session": {"gpu": int(gpu), "mixed_precision": False, "in_md5": False}}
model = BOW2ITM(mdl_cfg, dataset)
print(f"MD5={model.MD5}", flush=True)

if mode == "dev":
    model.train(dev=True, save_model=True)

elif mode == "final":
    model.train(dev=False, save_model=True)

elif mode == "eval":
    # Un proceso nuevo parte de pesos sin entrenar: hay que cargar los pesos finales
    # explicitamente antes de evaluar.
    model.train(dev=False, save_model=True)
    # evaluate()/predict() usan CONFIG["model"]["batch_size"] (el de ENTRENAMIENTO,
    # p.ej. 8192 para Amazon) para el batching interno. Con umbrales bajos de
    # min_reviews el numero de items (n_items) puede crecer muy por encima de lo
    # habitual (Amazon Fashion: 731 -> 27694 items a min_reviews=5), y algunas
    # metricas full-catalog (Recall@50, NDCG@100, etc.) construyen tensores
    # one-hot de tamaño batch x n_items (o mas), causando OOM en GPU con el
    # batch de entrenamiento. Reducir el batch SOLO para evaluar es seguro: no
    # afecta a los pesos ya entrenados ni a las rutas (MODEL_PATH ya fue fijado
    # en __init__), solo al tamaño de los tensores intermedios en evaluate/predict.
    if n_items > 5000:
        # El peor caso visto (Recall@50 con broadcast_weights) usa un tensor
        # ~ batch x 50 x n_items x 4 bytes; varias metricas full-catalog pueden
        # coexistir sin liberarse dentro de la misma llamada a evaluate(), asi
        # que se deja un margen generoso (presupuesto ~50MB por tensor de ese
        # tipo) en vez de calcular el batch "justo".
        eval_batch = max(8, int(50e6 / (50 * n_items * 4)))
        print(f"n_items={n_items} es grande; reduciendo batch de evaluacion de "
              f"{model.CONFIG['model']['batch_size']} a {eval_batch} para evitar OOM", flush=True)
        model.CONFIG["model"]["batch_size"] = eval_batch
    ret = model.evaluate(test=True, user_info=True, verbose=True)
    user_agg = pd.read_csv(f"{model.MODEL_PATH}final_user_agg_eval.csv")
    log_data = pd.read_csv(model.MODEL_PATH + "log.csv")

    row = {
        "dataset": dataset_name, "subset": subset_name,
        "min_reviews": min_reviews,
        "n_items": n_items,
        "n_train_dev": n_train_dev,
        "n_test": n_test,
        "epochs": int(log_data["epoch"].max() + 1),
        "ndcg10_review": float(ret["NDCG@10"].values[0]),
        "recall10_review": float(ret["Recall@10"].values[0]),
        "ndcg10_user": float(user_agg["NDCG@10_user"].mean()),
        "recall10_user": float(user_agg["Recall@10_user"].mean()),
        "md5": model.MD5,
    }
    print(json.dumps(row), flush=True)

    out_path = "results/min_reviews_ablation_results.csv"
    rows = pd.read_csv(out_path).to_dict("records") if os.path.exists(out_path) else []
    rows = [r for r in rows if not (r["dataset"] == dataset_name and r["subset"] == subset_name and r["min_reviews"] == min_reviews)] + [row]
    pd.DataFrame(rows).sort_values(["dataset", "subset", "min_reviews"]).to_csv(out_path, index=False)
    print(f"Guardado en {out_path}", flush=True)

else:
    raise ValueError(f"modo desconocido: {mode}")
