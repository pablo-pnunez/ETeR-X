# -*- coding: utf-8 -*-
from src.Common import get_pickle, print_w
from src.config import canonical_for_md5

import os
import json
import hashlib
import collections


class DatasetClass:

    def __init__(self, config, load):
        self.CONFIG = dict(collections.OrderedDict(sorted(config.items())))  # Sort to avoid spurious MD5 changes
        self.DATASET_PATH = self.CONFIG["save_path"] + self.__class__.__name__ + "/" + self.__get_md5__() + "/"

        # Create the dataset's folder
        if not os.path.exists(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)

            # Write the model configuration to a json file
            with open(self.DATASET_PATH + '/cfg.json', 'w') as fp:
                json.dump(self.CONFIG, fp, indent=4)

        if load is None: self.DATA = self.get_data()
        else: self.DATA = self.get_data(load=load)

    def get_data(self):
        """Returns a dictionary with the data"""
        raise NotImplementedError

    def get_dict_data(self, file_path, load):
        """Looks for the files in 'load' under 'file_path'"""
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
        """Gets an MD5 hash from the dataset configuration"""
        return hashlib.md5(str(canonical_for_md5(self.CONFIG)).encode()).hexdigest()
