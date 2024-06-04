# -*- coding: utf-8 -*-
from src.Common import get_pickle, print_w

import os
import json
import hashlib
import collections


class DatasetClass:

    def __init__(self, config, load):
        self.CONFIG = dict(collections.OrderedDict(sorted(config.items())))  # Ordenar para evitar cambios en MD5
        self.DATASET_PATH = self.CONFIG["save_path"] + self.__class__.__name__ + "/" + self.__get_md5__() + "/"

        # Crear carpeta para el dataset
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)

            # Crear json con la configuración del modelo
            with open(self.DATASET_PATH + '/cfg.json', 'w') as fp:
                json.dump(self.CONFIG, fp, indent=4)

        if load is None: self.DATA = self.get_data()
        else: self.DATA = self.get_data(load=load)

    def get_data(self):
        """Retorna un diccionario con los datos"""
        raise NotImplementedError

    def get_dict_data(self, file_path, load):
        """Busca en 'file_path' los ficheros en 'load'"""
        ret_dict = {}
        # a[["seq","rating","id_restaurant", "dev"]]
        for d in load:
            if os.path.exists(file_path + d):
                ret_dict[d] = get_pickle(file_path, d)

        if len(ret_dict) == len(load):
            return ret_dict
        else:
            return False

    def __get_md5__(self):
        """Obtiene un md5 a partir de la configuración del dataset"""
        return hashlib.md5(str(self.CONFIG).encode()).hexdigest()
