# -*- coding: utf-8 -*-
from src.Common import print_w

import os
import json
import random
import hashlib
import warnings
import numpy as np


class ModelClass:

    def __init__(self, config, dataset):
        self.CONFIG = config
        self.DATASET = dataset

        self.MODEL_NAME = self.__class__.__name__
        # self.CUSTOM_PATH = self.MODEL_NAME+"/"+self.CONFIG["id"]+"/"

        # MD5 without the gpu field so that it doesn't change
        self.MD5, cfg_save_data = self.__get_md5__()

        self.CUSTOM_PATH = self.MODEL_NAME+"/"+self.DATASET.CONFIG["dataset"]+"/"+self.DATASET.CONFIG["subset"]+"/"+self.MD5+"/"

        self.MODEL_PATH = "models/"+self.CUSTOM_PATH
        self.LOG_PATH = "logs/"+self.CUSTOM_PATH

        # Create the model's folder
        if os.path.exists(self.MODEL_PATH):
            print_w("Model folder already exists...")
        else:
            os.makedirs(self.MODEL_PATH, exist_ok=True)

        # Write the model configuration to a json file
        # with open(self.MODEL_PATH+'/cfg.json', 'w') as fp: json.dump(self.CONFIG["model"], fp)
        with open(self.MODEL_PATH+'/cfg.json', 'w') as fp:
            json.dump(cfg_save_data, fp, indent=4)

        # Select the most suitable GPU and cap its memory usage
        self.__config_session__(mixed_precision=self.CONFIG["session"]["mixed_precision"])

        # Fix the random seeds
        self.__fix_seeds__()

        # Build the model
        self.MODEL = self.get_model()

    def __get_md5__(self):
        # Gets the parts of the configuration dict that can alter a model (e.g. the gpu doesn't,
        # but the user percentage does). Returns the MD5.
        res = {"dataset_config": self.DATASET.CONFIG}
        for k in self.CONFIG.keys():
            if ("in_md5" not in self.CONFIG[k].keys()) or ("in_md5" in self.CONFIG[k].keys() and self.CONFIG[k]["in_md5"]):
                res[k] = self.CONFIG[k]
        return hashlib.md5(str(res).encode()).hexdigest(), res

    def __config_session__(self, mixed_precision):
        # Selects one of the available GPUs
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.CONFIG["session"]["gpu"])

    def __fix_seeds__(self):
        # Fix the numpy and TF random seeds
        np.random.seed(self.CONFIG["model"]["seed"])
        random.seed(self.CONFIG["model"]["seed"])

    def get_model(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError
