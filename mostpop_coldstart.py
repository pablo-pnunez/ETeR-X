# -*- coding: utf-8 -*-
"""
Numeros reales de MostPop bajo el protocolo cold-start (usuarios desconocidos
en TEST), para completar la paridad de los 7 baselines user-identity-agnostic
mencionados en Appendix B / Section~\ref{sec:results_b} de Main.tex.

MOSTPOP2ITM no tiene pesos que cargar (su "modelo" es un vector de popularidad
fijo, calculado directamente de TRAIN_DEV en get_model(), ver
src/models/text_models/MOSTPOP2ITM.py): basta construirlo sobre el dataset ya
cargado, sin necesidad de train()/checkpoint, igual que hace
run_one_coldstart_eval.py para BOW2ITM/USEM2ITM/BERT2ITM (pero ese script no
soporta MOSTPOP2ITM). Reporta, para cada dataset, tanto NDCG@10/Recall@10 sobre
el TEST completo ("known", sin filtrar -- misma definicion que usan
knn_bow_baseline.py/embedding_baseline.py sin --coldstart) como sobre el TEST
filtrado a usuarios ausentes en TRAIN_DEV ("coldstart", filtro identico a
Main.py known_users=False / run_one_coldstart_eval.py).

Uso: python mostpop_coldstart.py [gpu]
Salida: results/mostpop_coldstart_results.csv
"""
import json
import os
import sys

gpu = sys.argv[1] if len(sys.argv) > 1 else "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
os.environ["KERAS_BACKEND"] = "tensorflow"

import pandas as pd
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.models.text_models.MOSTPOP2ITM import MOSTPOP2ITM

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

COMBOS = [
    ("amazon", "fashion"), ("amazon", "digital_music"),
    ("pois", "barcelona"), ("pois", "madrid"), ("pois", "newyorkcity"), ("pois", "paris"), ("pois", "london"),
    ("restaurants", "gijon"), ("restaurants", "barcelona"), ("restaurants", "madrid"),
    ("restaurants", "newyorkcity"), ("restaurants", "paris"),
]

best = pd.read_csv("models/best_models.csv")

rows = []
for dataset, subset in COMBOS:
    row = best[(best.dataset == dataset) & (best.subset == subset) & (best.model == "MOSTPOP2ITM")]
    assert len(row) == 1, f"{dataset}/{subset}: se esperaba 1 fila en best_models.csv, se encontraron {len(row)}"
    md5 = row.iloc[0]["model_md5"]
    with open(f"models/MOSTPOP2ITM/{dataset}/{subset}/{md5}/cfg.json") as f:
        cfg = json.load(f)

    text_dataset = DATASET_CLASSES[dataset](cfg["dataset_config"])
    n_test_known = len(text_dataset.DATA["TEST"])

    mdl_cfg = {"model": cfg["model"], "session": {"gpu": int(gpu), "mixed_precision": False, "in_md5": False}}
    model = MOSTPOP2ITM(mdl_cfg, text_dataset)
    assert model.MD5 == md5, f"MD5 no coincide para {dataset}/{subset}: {model.MD5} != {md5}"
    model.train(dev=False, save_model=True)

    # --- "known": TEST completo, sin filtrar (misma definicion que knn_bow_baseline.py/embedding_baseline.py sin --coldstart) ---
    ret_known = model.evaluate(test=True, user_info=False, verbose=False)

    # --- "coldstart": filtro identico a Main.py known_users=False / run_one_coldstart_eval.py ---
    train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][~text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep="last")
    n_test_cold = len(text_dataset.DATA["TEST"])
    ret_cold = model.evaluate(test=True, user_info=False, verbose=False)

    row_out = {
        "dataset": dataset, "subset": subset,
        "n_test_known": n_test_known, "n_test_cold": n_test_cold,
        "ndcg10_known": float(ret_known["NDCG@10"].values[0]),
        "recall10_known": float(ret_known["Recall@10"].values[0]),
        "ndcg10_cold": float(ret_cold["NDCG@10"].values[0]),
        "recall10_cold": float(ret_cold["Recall@10"].values[0]),
    }
    print(json.dumps(row_out), flush=True)
    rows.append(row_out)

df = pd.DataFrame(rows)
df.to_csv("results/mostpop_coldstart_results.csv", index=False)
print("\n=== Promedios (12 datasets) ===")
print(df[["ndcg10_known", "recall10_known", "ndcg10_cold", "recall10_cold"]].mean())
print("Guardado en results/mostpop_coldstart_results.csv")
