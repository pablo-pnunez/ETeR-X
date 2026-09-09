# -*- coding: utf-8 -*-
"""
Baseline kNN-BoW: recuperacion no parametrica sobre representaciones Bag-of-Words.

Metodologia (descrita en Main.tex, Section~\ref{sec:baselines}): cada item se
representa por un perfil BoW agregado y L1-normalizado, construido a partir de
sus reviews de TRAIN_DEV (usando el mismo vocabulario PoS-filtrado que
ETeR-X). Una query (review de test) se rankea por similitud coseno contra cada
perfil de item. No tiene parametros aprendidos, por lo que no requiere
entrenamiento ni busqueda de hiperparametros.

Reutiliza el pipeline de datos de ETeR-X (TextDataset/BOW2ITM): BOW_SEQUENCES
ya esta L1-normalizado por fila y alineado posicionalmente con TRAIN_DEV/TEST
via la columna "bow", por lo que nunca hace falta un lookup por reviewId. Esto
hace que este baseline sea inmune a la clase de bug de reviewId duplicados que
afecta a USEM2ITM/BERT2ITM en Amazon (ver docs/reviews/analisis_revisiones.md
y el comentario en BERT2ITM.__create_tfdata__/USEM2ITM.__create_tfdata__).

Uso: python knn_bow_baseline.py [--coldstart]
Salida: results/knn_bow_results.csv, o results/knn_bow_results_coldstart.csv con --coldstart
  (dataset, subset, n_test, ndcg10, recall10, ndcg10_user, recall10_user)

--coldstart restringe TEST a reviews de usuarios que NO aparecen en TRAIN_DEV,
el mismo filtro que usa run_one_coldstart_eval.py para ETeR-X/USEM/BERT
(identico al de Main.py con known_users=False). Puesto que este baseline nunca
consulta la identidad del usuario, la comparacion known-vs-coldstart sirve
para confirmar empiricamente esa invarianza en vez de asumirla.
"""
import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import ndcg_score

from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset

DATASET_CLASSES = {"restaurants": RestaurantDataset, "pois": POIDataset, "amazon": AmazonDataset}

COMBOS = [
    ("amazon", "fashion"), ("amazon", "digital_music"),
    ("pois", "barcelona"), ("pois", "madrid"), ("pois", "newyorkcity"), ("pois", "paris"), ("pois", "london"),
    ("restaurants", "gijon"), ("restaurants", "barcelona"), ("restaurants", "madrid"),
    ("restaurants", "newyorkcity"), ("restaurants", "paris"),
]

K = 10


def evaluate_dataset(dataset, subset, coldstart=False):
    # El MD5 del checkpoint de referencia se lee de models/best_models.csv en
    # vez de localizarlo con un glob sobre el directorio de checkpoints: mas
    # de un checkpoint puede compartir la misma configuracion de vocabulario
    # (pos_tagging_fix=true) cuando existen ademas checkpoints de ablaciones
    # (bow_pct_words, min_reviews_rst) para el mismo dataset/subset, asi que
    # un glob ya no identifica un unico candidato de forma fiable.
    best = pd.read_csv("models/best_models.csv")
    row = best[(best.dataset == dataset) & (best.subset == subset) & (best.model == "BOW2ITM")]
    assert len(row) == 1, f"{dataset}/{subset}: se esperaba 1 fila en best_models.csv, se encontraron {len(row)}"
    md5 = row.iloc[0]["model_md5"]
    with open(f"models/BOW2ITM/{dataset}/{subset}/{md5}/cfg.json") as f:
        cfg = json.load(f)

    ds = DATASET_CLASSES[dataset](cfg["dataset_config"])

    train_dev = ds.DATA["TRAIN_DEV"]
    test = ds.DATA["TEST"]

    if coldstart:
        # Filtro identico al de run_one_coldstart_eval.py / Main.py known_users=False
        train_dev_users = train_dev.userId.unique()
        test = test[~test["userId"].isin(train_dev_users)]
        test = test.drop_duplicates(subset=["userId", "id_item"], keep="last")
    bow_seq = ds.DATA["BOW_SEQUENCES"]  # ya L1-normalizado por fila, alineado por posicion via "bow"
    n_items = ds.DATA["N_ITEMS"]

    # --- Perfil de item: suma de los vectores BoW (ya L1-norm por review) de
    # sus reviews de TRAIN_DEV, renormalizado L1 ---
    # (agrupar una sola vez, en vez de filtrar train_dev por cada item -> O(n_items*n_train) evitado)
    idx_by_item = train_dev.groupby("id_item")["bow"].apply(lambda s: s.values).to_dict()
    vocab_size = bow_seq.shape[1]
    item_rows = []
    for item_id in range(n_items):
        idx = idx_by_item.get(item_id, np.array([], dtype=int))
        if len(idx) == 0:
            item_rows.append(bow_seq[0:1] * 0)  # item sin reviews en train (no deberia pasar tras el filtro de min_reviews)
            continue
        profile = bow_seq[idx].sum(axis=0)
        item_rows.append(np.asarray(profile).reshape(1, vocab_size))
    item_profiles = csr_matrix(np.vstack(item_rows))
    item_profiles = normalize(item_profiles, axis=1, norm="l1")

    # Para similitud coseno con vectores no-negativos, normalizar L2 y hacer producto punto
    item_profiles_l2 = normalize(item_profiles, axis=1, norm="l2")

    # --- Query: el propio vector BoW (ya L1-norm) de cada review de test ---
    query_idx = test["bow"].values
    queries = bow_seq[query_idx]
    queries_l2 = normalize(queries, axis=1, norm="l2")

    true_items = test["id_item"].values

    # --- Preparar acumuladores por-usuario: agregar por usuario, no por
    # review, para ser homogeneo con los baselines CF de Cornac (que evaluan
    # una unica prediccion por usuario, no una por interaccion) ---
    user_ids = test["userId"].values
    unique_users, user_row = np.unique(user_ids, return_inverse=True)
    n_users = len(unique_users)
    user_score_sum = np.zeros((n_users, n_items), dtype=np.float32)
    user_review_count = np.zeros(n_users, dtype=np.int32)
    user_relevant = [set() for _ in range(n_users)]

    # Batch para no materializar una matriz densa (n_test x n_items) entera de golpe
    batch_size = 4096
    ndcgs, hits = [], []
    for start in range(0, queries_l2.shape[0], batch_size):
        q_batch = queries_l2[start:start + batch_size]
        scores = (q_batch @ item_profiles_l2.T).toarray()  # (batch, n_items)
        true_batch = true_items[start:start + batch_size]
        urow_batch = user_row[start:start + batch_size]

        true_scores = scores[np.arange(len(true_batch)), true_batch]
        ranks = (scores > true_scores[:, None]).sum(axis=1) + 1  # ties a favor del item real
        hit = ranks <= K
        ndcg = np.where(hit, 1.0 / np.log2(ranks + 1), 0.0)

        ndcgs.append(ndcg)
        hits.append(hit)

        np.add.at(user_score_sum, urow_batch, scores)
        np.add.at(user_review_count, urow_batch, 1)
        for ur, it in zip(urow_batch, true_batch):
            user_relevant[ur].add(int(it))

    ndcgs = np.concatenate(ndcgs)
    hits = np.concatenate(hits)

    # --- Agregacion por-usuario: media de los vectores de score de sus
    # reviews de test, union de sus items relevantes (mismo criterio que
    # RSTModel.__per_user_ranking_eval__) ---
    user_ndcgs, user_hits = [], []
    for u in range(n_users):
        relevant = user_relevant[u]
        if not relevant:
            continue
        mean_score = user_score_sum[u] / user_review_count[u]
        top_k = np.argsort(-mean_score)[:K]
        n_hit = len(set(top_k.tolist()) & relevant)
        user_hits.append(n_hit / len(relevant))

        relevant_vec = np.zeros(n_items, dtype=np.int32)
        relevant_vec[list(relevant)] = 1
        user_ndcgs.append(ndcg_score([relevant_vec], [mean_score], k=K))

    return {
        "dataset": dataset, "subset": subset, "n_test": len(test), "n_users": n_users,
        "ndcg10": float(ndcgs.mean()), "recall10": float(hits.mean()),
        "ndcg10_user": float(np.mean(user_ndcgs)), "recall10_user": float(np.mean(user_hits)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coldstart", action="store_true")
    args = parser.parse_args()

    out_path = "results/knn_bow_results_coldstart.csv" if args.coldstart else "results/knn_bow_results.csv"

    rows = []
    for dataset, subset in COMBOS:
        print(f"=== {dataset}/{subset} ({'coldstart' if args.coldstart else 'known users'}) ===", flush=True)
        res = evaluate_dataset(dataset, subset, coldstart=args.coldstart)
        print(f"  n_test={res['n_test']} n_users={res['n_users']} "
              f"NDCG@10={res['ndcg10']:.4f} Recall@10={res['recall10']:.4f} | "
              f"NDCG@10_user={res['ndcg10_user']:.4f} Recall@10_user={res['recall10_user']:.4f}", flush=True)
        rows.append(res)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Guardado en {out_path}")
