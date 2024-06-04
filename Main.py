# -*- coding: utf-8 -*-
from src.Common import parse_cmd_args, print_b, print_e
import pandas as pd
import numpy as np
import nvgpu
import json
import os

args = parse_cmd_args()
gpu = np.argmin([g["mem_used_percent"] for g in nvgpu.gpu_info()]) if args.gpu is None else args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
os.environ["KERAS_BACKEND"] = "tensorflow" 

from src.datasets.text_datasets.RestaurantDataset import RestaurantDataset
from src.datasets.text_datasets.AmazonDataset import AmazonDataset
from src.datasets.text_datasets.POIDataset import POIDataset

from src.models.text_models.att.ATT2ITM import ATT2ITM
from src.models.text_models.att.ATT2ITM_2 import ATT2ITM_2 # como el normal, pero sin tanh.

from src.models.text_models.BOW2ITM import BOW2ITM
from src.models.text_models.USEM2ITM import USEM2ITM
from src.models.text_models.BERT2ITM import BERT2ITM

from src.models.text_models.MOSTPOP2ITM import MOSTPOP2ITM

# #######################################################################################################################


model = "MOSTPOP2ITM" if args.mn is None else args.mn
dataset = "restaurants".lower().replace(" ", "") if args.dst is None else args.dst
subset = "gijon".lower().replace(" ", "") if args.sst is None else args.sst

# from src.experiments.Common import load_best_model

# model = load_best_model(model=model, dataset=dataset, subset=subset)
# model.train(dev=False, save_model=True)
# model.evaluate_text("Where can i eat the best vegan burger")

# exit()

# stage = 3 if args.stg is None else args.stg
# use_best = True

stage = -1 if args.stg is None else args.stg
use_best = False

if use_best:  # Usar la mejor configuración conocida?
    best_model = pd.read_csv("models/best_models.csv")
    best_model = best_model.loc[(best_model.dataset == dataset) & (best_model.subset == subset) & (best_model.model == model)]["model_md5"].values[0]
    # best_model = "1badd2185814e515e43609bc6b2c13ae"
    model_path = f"models/{model}/{dataset}/{subset}/{best_model}"
    with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
    dts_cfg = model_config["dataset_config"]
    with open(f'{model_path}/cfg.json') as f: model_config = json.load(f)
    mdl_cfg = {"model": model_config["model"], "session": {"gpu": gpu, "mixed_precision": True if model_config["model"]["model_version"]=="2" else False, "in_md5": False}}

    print_b(f"Loading best model:{best_model}")

else:
    model_v = "0" if args.mv is None else args.mv
    neg_rate = 10

    seed = 100 if args.sd is None else args.sd
    l_rate = 1e-4 if args.lr is None else args.lr
    n_epochs = 1000 if args.ep is None else args.eps
    b_size = 256 if args.bs is None else args.bs
    early_stop_patience = 50 if args.esp is None else args.esp

    # if city=="london": b_size = 512

    min_reviews_rst = 100
    min_reviews_usr = 1
    bow_pct_words = 10 if args.bownws is None else args.bownws
    w2v_dimen = 512  # 300

    remove_stopwords = 2  # 0, 1 o 2 (No quitar, quitar manual, quitar automático)
    lemmatization = True
    remove_accents = True
    remove_numbers = True
    truncate_padding = True

    language = "es" if subset in ["gijon", "madrid", "barcelona"] else "fr" if subset in ["paris"] else "en"

    if dataset == "restaurants":
        base_path = "/media/nas/datasets/tripadvisor/restaurants/"
    elif dataset == "pois":
        base_path = "/media/nas/datasets/tripadvisor/pois/"
        language = "es"  # Están todas en español
    elif dataset == "amazon":
        base_path = "/media/nas/datasets/amazon/"

    # DATASET CONFIG #######################################################################################################

    dts_cfg = {"dataset": dataset, "subset": subset, "language": language, "seed": seed, "data_path": base_path, "save_path": "data/",  # base_path + "Datasets/",
               "remove_stopwords": remove_stopwords, "remove_accents": remove_accents, "remove_numbers": remove_numbers,
               "lemmatization": lemmatization,
               "min_reviews_rst": min_reviews_rst, "min_reviews_usr": min_reviews_usr,
               "min_df": 5, "bow_pct_words": bow_pct_words, "presencia": False, "text_column": "text",  # BOW
               "n_max_words": -50, "test_dev_split": .1, "truncate_padding": truncate_padding}

if dataset == "restaurants":
    # text_dataset = RestaurantDataset(dts_cfg, load=["TRAIN_DEV", "TEXT_TOKENIZER", "TEXT_SEQUENCES", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "N_ITEMS", "FEATURES_NAME", "BOW_SEQUENCES"])
    text_dataset = RestaurantDataset(dts_cfg)
elif dataset == "pois":
    text_dataset = POIDataset(dts_cfg)
elif dataset == "amazon":
    text_dataset = AmazonDataset(dts_cfg)
else:
    raise ValueError

known_users = True

if known_users is False:
    print_e("Se dejan solo usuarios desconocidos!!")
    # Se buscan los usuarios de train+dev y se eliminan de test
    train_dev_users = text_dataset.DATA["TRAIN_DEV"].userId.unique()
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"][~text_dataset.DATA["TEST"]["userId"].isin(train_dev_users)]
    text_dataset.DATA["TEST"] = text_dataset.DATA["TEST"].drop_duplicates(subset=["userId", "id_item"], keep='last', inplace=False)

if "MOSTPOP2ITM" == model:
    mostpop2itm_mdl = MOSTPOP2ITM({"model":{"batch_size": b_size, "learning_rate": l_rate, "seed":seed}, "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}, text_dataset)   
    mostpop2itm_mdl.train() # Crea un csv falso
    # mostpop2itm_mdl.evaluate()

    
elif "ATT2ITM_2" == model:

    if use_best:
        att2itm_mdl_cfg = mdl_cfg
    else:
        att2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                     "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": early_stop_patience},
                           "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

    att2itm_mdl = ATT2ITM_2(att2itm_mdl_cfg, text_dataset)

    if stage == 0:
        att2itm_mdl.train(dev=True, save_model=True)

    if stage == -1:
        att2itm_mdl.train(dev=True, save_model=False)
        att2itm_mdl.evaluate_text("Quiero comer un buen arroz con bogavante y con buenas vistas")

        att2itm_mdl.emb_tsne()

    if stage == 3:
        att2itm_mdl.train(dev=False, save_model=True)
        att2itm_mdl.evaluate_text("Where can i eat the typical pastrami sandwich")

elif "ATT2ITM" == model:

    if use_best:
        att2itm_mdl_cfg = mdl_cfg
    else:
        att2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                     "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": early_stop_patience},
                           "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

    att2itm_mdl = ATT2ITM(att2itm_mdl_cfg, text_dataset)

    if stage == 0:
        att2itm_mdl.train(dev=True, save_model=True)

    if stage == -1:
        att2itm_mdl.train(dev=True, save_model=False)
        att2itm_mdl.evaluate_text("Quiero comer un arroz con bogavante y con buenas vistas")
        # att2itm_mdl.evaluate_text("imagino que me interesa profundamente la definitivas")
        # att2itm_mdl.all_words_analysis()
        # att2itm_mdl.emb_tsne()

    if stage == -2:
        att2itm_mdl.train(dev=True, save_model=True)
        # att2itm_mdl.evaluate()
        att2itm_mdl.evaluate(test=True)

        # att2itm_mdl.evaluate_text("Quiero comer un buen arroz con bogavante y con buenas vistas")
        # att2itm_mdl.evaluate_text("Quiero arroz con bogavante con nutella")
        # att2itm_mdl.evaluate_text("Je veux manger du riz avec du homard et avec une belle vue")
        # att2itm_mdl.evaluate_text("I want a black and red and also white leather wallet for my kid")
        # att2itm_mdl.all_words_analysis()
        # att2itm_mdl.emb_tsne()
        exit()

        # att2itm_mdl.evaluate_text("Quiero arroz con bogavante y nutella")

        # OJO: SELECCIONAR PALABRAS DE QUERY EN FUNCIÓN DE TODAS LAS PALABRAS DEL VOACABULARIO (NO SOLO LAS DE LA CONSULTA)

        # att2itm_mdl.evaluate_text("I want fresh pizza")

        att2itm_mdl.evaluate_text("Quiero ir al kausa")
        att2itm_mdl.evaluate_text("Estrella michelin con vistas al mar")

        # att2itm_mdl.evaluate_text("cheap pizza")
        # att2itm_mdl.evaluate_text("I want the last album of ed sheeran")

        # att2itm_mdl.evaluate_text("I want pastrami sandwich")

        # att2itm_mdl.evaluate_text("Quiero comer un arroz con bogavante y con buenas vistas")

        # att2itm_mdl.evaluate_text("Quiero comer una hamburguesa cara")
        # att2itm_mdl.evaluate_text("hamburguesa cara")

        # att2itm_mdl.emb_tsne()

        exit()

        test_real_sample = pd.read_pickle(text_dataset.DATASET_PATH+"ALL_DATA")
        test_real_sample = test_real_sample[test_real_sample.test==1].iloc[10]
        att2itm_mdl.evaluate_text(test_real_sample.text_source)
        print("[REAL ↑]: ", test_real_sample["name"])
        

        if language == "es":
            if subset == "gijon":
                # att2itm_mdl.word_analysis("interesar")
                # att2itm_mdl.word_analysis("pizza")
                # att2itm_mdl.word_analysis("imagino")
                # att2itm_mdl.word_analysis("definitiva")
                # att2itm_mdl.word_analysis("profundamente")            
                att2itm_mdl.evaluate_text("Quiero comer un arroz con bogavante y con buenas vistas")

            elif subset == "barcelona":
                # att2itm_mdl.word_analysis("albahaca")
                # att2itm_mdl.word_analysis("suponer")
                # att2itm_mdl.word_analysis("caso")
                # att2itm_mdl.word_analysis("descontar")
                att2itm_mdl.evaluate_text("Quiero comer un arroz con bogavante y con buenas vistas")
                att2itm_mdl.evaluate_text("El caso es suponer no descontar la albahaca")
            
            elif subset == "madrid":
                att2itm_mdl.evaluate_text("Quiero comer un arroz con bogavante y con buenas vistas")
                att2itm_mdl.evaluate_text("El caso es suponer no descontar la albahaca")

        elif language == "en":
            # att2itm_mdl.evaluate_text("Where can i eat the typical pastrami sandwich")
            att2itm_mdl.evaluate_text("I want the cheapest digital keyboard")

    if stage == 3:
        att2itm_mdl.train(dev=False, save_model=True)
        # att2itm_mdl.evaluate(test=True)
        att2itm_mdl.evaluate_text("Where can i eat the typical pastrami sandwich")

elif "BOW2ITM" == model:

    if use_best:
        bow2itm_mdl_cfg = mdl_cfg
    else:
        bow2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                    "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 50},
                        "session": {"gpu": gpu, "mixed_precision": False, "in_md5": False}}

    bow2itm_mdl = BOW2ITM(bow2itm_mdl_cfg, text_dataset)

    if stage == 0:
        bow2itm_mdl.train(dev=True, save_model=True)

    if stage == -1:
        bow2itm_mdl.train(dev=True, save_model=False)

    if stage == -2:
        bow2itm_mdl.train(dev=True, save_model=True)

    if stage == 3:
        bow2itm_mdl.train(dev=False, save_model=True)
        bow2itm_mdl.evaluate(test=True)

elif "USEM2ITM" == model:

    if use_best:
        usem2itm_mdl_cfg = mdl_cfg
    else:
        usem2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                  "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 50},
                        "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

    if stage == 0:
        usem2itm_mdl = USEM2ITM(usem2itm_mdl_cfg, text_dataset)
        usem2itm_mdl.train(dev=True, save_model=True)

    if stage == -1:
        usem2itm_mdl = USEM2ITM(usem2itm_mdl_cfg, text_dataset)
        usem2itm_mdl.train(dev=True, save_model=False)

elif "BERT2ITM" == model:

    if use_best:
        bert2itm_mdl_cfg = mdl_cfg
    else:
        bert2itm_mdl_cfg = {"model": {"model_version": model_v, "learning_rate": l_rate, "final_learning_rate": l_rate/100, "epochs": n_epochs, "batch_size": b_size, "seed": seed,
                                  "early_st_first_epoch": 0, "early_st_monitor": "val_loss", "early_st_monitor_mode": "min", "early_st_patience": 50},
                        "session": {"gpu": gpu, "mixed_precision": True, "in_md5": False}}

    if stage == 0:
        bert2itm_mdl = BERT2ITM(bert2itm_mdl_cfg, text_dataset)
        bert2itm_mdl.train(dev=True, save_model=True)

    if stage == -1:
        bert2itm_mdl = BERT2ITM(bert2itm_mdl_cfg, text_dataset)
        bert2itm_mdl.train(dev=True, save_model=False)

    if stage == 3:
        bert2itm_mdl = BERT2ITM(bert2itm_mdl_cfg, text_dataset)
        bert2itm_mdl.train(dev=False, save_model=True)
        bert2itm_mdl.evaluate(test=True, user_info=True)