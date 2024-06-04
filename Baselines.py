import numpy as np
import argparse
import nvgpu
import os

parser = argparse.ArgumentParser()
parser.add_argument('-dst', type=str, help="Dataset")
parser.add_argument('-sst', type=str, help="Subset")
parser.add_argument('-gpu', type=int, help="GPU")
args = parser.parse_args()

gpu =  int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info())))) if args.gpu is None else args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

print(f"USING GPU-{gpu}.")

from src.Common import print_b, print_e
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.att.ATT2ITM import ATT2ITM
from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

from cornac.metrics import Recall, Precision, FMeasure, NDCG
from cornac.hyperopt import GridSearch, Discrete
from cornac.eval_methods import BaseMethod
from cornac.data.text import BaseTokenizer
from cornac.experiment import Experiment
from cornac.data import ReviewModality
import cornac

import tensorflow as tf
import pandas as pd
import json


def load_set(dataset, subset):

    models = ["USEM2ITM", "BERT2ITM", "ATT2ITM", "BOW2ITM"]
    eval_data = {}

    for model in models:
        best_model = pd.read_csv("models/best_models.csv")
        best_model = best_model.loc[(best_model.dataset == dataset) & (best_model.subset == subset) & (best_model.model == model)]["model_md5"].values[0]
        model_path = f"models/{model}/{dataset}/{subset}/{best_model}"
        with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
        dts_cfg = model_config["dataset_config"]
        with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
        mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": False, "in_md5": False}}

        # print_b(f"Loading best model: {best_model}")

        if dataset == "restaurants":
            # text_dataset = RestaurantDataset(dts_cfg, load=["TRAIN_DEV", "TEXT_TOKENIZER", "TEXT_SEQUENCES", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "N_ITEMS", "FEATURES_NAME", "BOW_SEQUENCES"])
            text_dataset = RestaurantDataset(dts_cfg)
        elif dataset == "pois":
            text_dataset = POIDataset(dts_cfg)
        elif dataset == "amazon":
            text_dataset = AmazonDataset(dts_cfg)
        else:
            raise ValueError

        # Primero se eliminan los usuarios que no están en train de los conjuntos de val y test
        # Esto es solo para evaluar nuestro modelo en igualdad de condiciones
        # train_users = text_dataset.DATA["TRAIN_DEV"][text_dataset.DATA["TRAIN_DEV"]["dev"] == 0].userId.unique()
        train_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
        text_dataset.DATA["TRAIN_DEV"] = text_dataset.DATA["TRAIN_DEV"][text_dataset.DATA["TRAIN_DEV"]["userId"].isin(train_users)]
        text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][text_dataset.DATA["TEST"]["userId"].isin(train_users)]
        text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)

        # Evaluar nuestro modelo
        if "ATT2ITM" == model: model_class = ATT2ITM(mdl_cfg, text_dataset)
        elif "BOW2ITM" == model: model_class = BOW2ITM(mdl_cfg, text_dataset)
        elif "USEM2ITM" == model: model_class = USEM2ITM(mdl_cfg, text_dataset)
        elif "BERT2ITM" == model: model_class = BERT2ITM(mdl_cfg, text_dataset)
        else: raise NotImplementedError    
        # Cargar el modelo y evaluar
        model_class.train(dev=False, save_model=True)
        eval_data[model] = model_class.evaluate(test=True)

    # Número de reseñas de cada usuario en train/dev (para separar resultados por usuario)
    train_dev_user_count = text_dataset.DATA["TRAIN_DEV"].groupby("userId").agg(cold=("userId","count")).reset_index()

    # Luego se cargan los datos y se adaptan a Cornac
    all_data = pd.read_pickle(f"{text_dataset.DATASET_PATH}ALL_DATA")
    all_data["rating"] /= 10
    all_data = all_data[["userId", "id_item", "rating", "dev", "test", "text"]]

    # Eliminar usuarios desconocidos y dividir en 3 subconjuntos
    train_data = all_data[(all_data["dev"] == 0) & (all_data["test"] == 0)]
    train_users = train_data["userId"].unique()
    id_user, _ = pd.factorize(train_data["userId"])
    # user_map = pd.DataFrame(zip(train_data["userId"], id_user), columns=["userId", "id_user"])
    val_data = all_data[(all_data["dev"] == 1) & (all_data["userId"].isin(train_users))]
    test_data = all_data[(all_data["test"] == 1) & (all_data["userId"].isin(train_users))]

    # train_data = train_data.merge(user_map)[["id_user", "id_item", "rating"]]
    # val_data = val_data.merge(user_map)[["id_user", "id_item", "rating"]].drop_duplicates(subset=["id_user", "id_item"], keep='last', inplace=False)
    # test_data = test_data.merge(user_map)[["id_user", "id_item", "rating"]].drop_duplicates(subset=["id_user", "id_item"], keep='last', inplace=False)
    
    train_data = train_data[["userId", "id_item", "rating"]]
    val_data = val_data[["userId", "id_item", "rating"]].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)
    test_data = test_data[["userId", "id_item", "rating"]].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)

    # print_e(f"TEST USERS: {len(test_data['userId'].unique())}")

    # Instantiate a Base evaluation method using the provided train and test sets
    eval_method = BaseMethod.from_splits(train_data=train_data.to_records(index=False), val_data=val_data.to_records(index=False), test_data=test_data.to_records(index=False),  verbose=False, rating_threshold=1)
    # OJO: lo anterior elimina las repeticiones de USUARIO, ITEM
    # OJO: Si se pone un rating treshold, en test, por algún motivo se eliminan los menores de este. Con uno no pasa nada.

    return eval_data, eval_method, train_dev_user_count


seed = 2048
base_path = "models/Baselines"

# datasets = {"pois": ["barcelona", "gijon"]}
dataset = "restaurants" if args.dst is None else args.dst
subset = "gijon" if args.sst is None else args.sst

# Cargar los datos
eval_data, eval_method, train_dev_user_count = load_set(dataset, subset)

user_id_map = pd.DataFrame(eval_method.test_set.uid_map.items(), columns=["userId", "id_user"])

metrics = [
    FMeasure(), FMeasure(k=1), FMeasure(k=5), FMeasure(k=10),
    Recall(), Recall(k=1), Recall(k=5), Recall(k=10), Recall(k=20), Recall(k=50),
    Precision(), Precision(k=1), Precision(k=5), Precision(k=10),
    NDCG(), NDCG(k=1), NDCG(k=10), NDCG(k=50), NDCG(k=100),
    ]


md_bpr = cornac.models.BPR(seed=seed, verbose=True)
md_ease = cornac.models.EASE(seed=seed, verbose=True)

models = [
    cornac.models.MostPop(name="MostPop"),
    cornac.models.MF(use_bias=True),
    cornac.models.OnlineIBPR(),
    #cornac.models.BiVAECF(k=20, encoder_structure=[40], n_epochs=500, batch_size=128), # Con parámetros que aparecen en gitgub, pero va igual de mal
    GridSearch( model=md_bpr, space=[ Discrete("k", [25, 50]), Discrete("max_iter", [50, 100]), Discrete("learning_rate", [1e-4, 5e-4, 1e-3]), ], metric=NDCG(), eval_method=eval_method),
    GridSearch( model=md_ease, space=[ Discrete("posB", [True, False]), ], metric=NDCG(), eval_method=eval_method), # cornac.models.MF(seed=seed),  # Best parameter settings: {'k': 30, 'learning_rate': 5e-06, 'max_iter': 10}
    cornac.models.BiVAECF(batch_size=(64 if dataset=="restaurants" and subset=="paris" else 100)),

    # cornac.models.MMMF(seed=seed),  # Best parameter settings: {'k': 5, 'learning_rate': 0.001, 'max_iter': 50}
    # cornac.models.NeuMF(seed=seed),
    # cornac.models.WBPR(seed=seed),
    # cornac.models.FM(seed=seed),
    # cornac.models.HPF(seed=seed),
    # cornac.models.NMF(seed=seed),
    # cornac.models.PMF(seed=seed),
    # cornac.models.SKMeans(seed=seed),
    # cornac.models.SVD(seed=seed),
    # cornac.models.WMF(seed=seed),
]

experiment = Experiment(
    eval_method=eval_method,
    show_validation=True,
    models=models,
    metrics=metrics,
    save_dir=f"{base_path}/{dataset}/{subset}",
    verbose=True,
    user_based=False,
)

experiment.run()

metric_names = [m.name for m in experiment.metrics]
model_names = []
final_res = []

for model_name, model_data in eval_data.items():
    model_names.append(model_name)
    final_res.append(model_data[metric_names].values.tolist()[0])

user_final_res = None
for result in experiment.result:
    model_names.append(result.model_name)
    final_res.append([result.metric_avg_results[mtr] for mtr in metric_names])

    # Separar por usuario
    user_metric = "NDCG@-1"   
    usr_results = pd.DataFrame(result.metric_user_results).reset_index().rename(columns={"index":"id_user"}).merge(user_id_map, how="left")
    usr_results = usr_results.merge(train_dev_user_count, how="left")
    usr_results = usr_results[["userId", "cold", user_metric]].rename(columns={user_metric: result.model_name})

    if user_final_res is None: user_final_res = usr_results
    else: user_final_res = user_final_res.merge(usr_results, how="left")

final_res = pd.DataFrame(final_res, columns=metric_names)
final_res.insert(0, "Model", model_names)
final_res = final_res.sort_values("Model")
final_res.to_csv(f"{base_path}/{dataset}/{subset}/results.csv", index=False)

user_final_res[["userId", "cold"]+[r.model_name for r in experiment.result]].to_csv(f"{base_path}/{dataset}/{subset}/user_results.csv", index=False)

print(final_res.to_string())
