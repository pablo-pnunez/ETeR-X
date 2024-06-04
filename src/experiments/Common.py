# -*- coding: utf-8 -*-
from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.BERT2ITM import BERT2ITM
from src.models.text_models.MOSTPOP2ITM import MOSTPOP2ITM

from src.Common import print_w, print_b

import pandas as pd
import numpy as np
import nvgpu
import json

def load_best_config(model, dataset, subset, gpu=None, base_path=""):
    """Carga y retorna la mejor configuración de modelo y dataset según los parámetros datos. Se carga desde el fichero 'best_models.csv'
    Args:
        model (str, required): Nombre del modelo a cargar. 
        dataset (str, required): Nombre del conjunto de datos.
        subset (str, required): Nombre del subconjunto de datos.
        gpu (int, optional): Entero indicando GPU.
    Returns:
        dts_cfg, mdl_cfg: Mejores configuraciones
    """
    # ToDo: Cambiar esto para que no se lea del fichero, se obtenga directamente de la carpeta models mirando aquellos cuyo entrenamiento finalizó
    # OJO: Lo anterior la puede liar si tengo ya modelos finales y me pongo a hacer pruebas y una de ellas mejora lo existente.

    assert model is not None and dataset is not None and subset is not None

    if gpu is None: gpu = int(np.argmin(list(map(lambda x: x["mem_used_percent"], nvgpu.gpu_info())))) 

    best_model = pd.read_csv(base_path+"models/best_models.csv")
    best_model = best_model.loc[(best_model.dataset == dataset) & (best_model.subset == subset) & (best_model.model == model)]["model_md5"].values[0]
    model_path = f"{base_path}models/{model}/{dataset}/{subset}/{best_model}"
    with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
    dts_cfg = model_config["dataset_config"]
    with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
    mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": False, "in_md5": False}}

    print_b(f"Loading best {model} model: {best_model}")

    return dts_cfg, mdl_cfg

def load_best_model(model, dataset, subset, gpu=None, base_path=""):
    """Retorna la mejor combinación de parámetros para el modelo-dataset dados.
    Args:
        model (str, required): Nombre del modelo a cargar. 
        dataset (str, required): Nombre del conjunto de datos.
        subset (str, required): Nombre del subconjunto de datos.
        gpu (int, optional): Entero indicando GPU.
    Returns:
        model_class: Modelo con la mejor combinación de parámetros existente.
    """
    
    dts_cfg, mdl_cfg = load_best_config(model=model, dataset=dataset, subset=subset, gpu=gpu, base_path=base_path)

    if dataset == "restaurants": text_dataset = RestaurantDataset(dts_cfg)
    elif dataset == "pois": text_dataset = POIDataset(dts_cfg)
    elif dataset == "amazon": text_dataset = AmazonDataset(dts_cfg)
    else: raise ValueError
    
    if "BOW2ITM" == model: model_class = BOW2ITM(mdl_cfg, text_dataset)
    elif "USEM2ITM" == model: model_class = USEM2ITM(mdl_cfg, text_dataset)
    elif "BERT2ITM" == model: model_class = BERT2ITM(mdl_cfg, text_dataset)
    elif "MOSTPOP2ITM" == model: model_class = MOSTPOP2ITM(mdl_cfg, text_dataset)
    else: raise ValueError    

    print_w("Model weights are not loaded!")

    return model_class