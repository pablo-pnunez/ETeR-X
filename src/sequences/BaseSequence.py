# -*- coding: utf-8 -*-
from tensorflow.keras.utils import Sequence
import numpy as np


class BaseSequenceXY(Sequence):
    """ Esqueleto de una secuencia."""

    def __init__(self, parent_model):

        self.MODEL = parent_model  # Modelo de keras que contiene todos los datos de configuración
        self.BATCH_SIZE = self.MODEL.CONFIG["model"]["batch_size"]
        self.MODEL_DATA = self.init_data()

        self.X = np.array_split(self.MODEL_DATA[0], len(self.MODEL_DATA[0]) // self.BATCH_SIZE)
        self.Y = np.array_split(self.MODEL_DATA[1], len(self.MODEL_DATA[1]) // self.BATCH_SIZE)

    def init_data(self) -> tuple:
        return NotImplementedError

    def preprocess_input(self, batch_data):
        return NotImplementedError

    def preprocess_output(self, batch_data):
        return NotImplementedError

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        batch_X = self.X[idx]
        batch_Y = self.Y[idx]

        input_data = self.preprocess_input(batch_X)
        output_data = self.preprocess_output(batch_Y)

        return input_data, output_data


class BaseSequence(Sequence):
    """ Esqueleto de una secuencia."""

    def __init__(self, parent_model):

        self.MODEL = parent_model  # Modelo de keras que contiene todos los datos de configuración
        self.BATCH_SIZE = self.MODEL.CONFIG["model"]["batch_size"]
        self.MODEL_DATA = self.init_data()

        if len(self.MODEL_DATA) > self.BATCH_SIZE:
            self.BATCHES = np.array_split(self.MODEL_DATA, len(self.MODEL_DATA) // self.BATCH_SIZE)
        else:
            self.BATCHES = np.array_split(self.MODEL_DATA, 1)

    def init_data(self):
        return NotImplementedError

    def preprocess_input(self, batch_data):
        return NotImplementedError

    def preprocess_output(self, batch_data):
        return NotImplementedError

    def __len__(self):
        return len(self.BATCHES)

    def __getitem__(self, idx):
        batch_data = self.BATCHES[idx]

        input_data = self.preprocess_input(batch_data)
        output_data = self.preprocess_output(batch_data)

        return input_data, output_data
    