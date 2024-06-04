# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
from codecarbon import EmissionsTracker
import tensorflow as tf
import pandas as pd
import numpy as np
import os

class MOSTPOP2ITM(RSTModel):
    """ Baseline """
    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        self.emissions_computed = os.path.exists(self.MODEL_PATH+"emissions_train.csv")
        # Medir emisiones        
        if not self.emissions_computed:
            self.emissions_tracker = EmissionsTracker(project_name="Epoch 0", log_level="error", output_dir=self.MODEL_PATH, output_file="emissions_train.csv", tracking_mode="process")
            self.emissions_tracker.start()
        # Número de reviews por user 
        item_pop = self.DATASET.DATA["TRAIN_DEV"]["id_item"].value_counts().reset_index().sort_values("index").reset_index(drop=True)["id_item"].values
        # Normalizar entre 0 y 1 (innecesario, pero bueno)
        most_pop = np.round((item_pop-min(item_pop))/(max(item_pop)-min(item_pop)),2).tolist()
        # Definir la entrada del modelo
        input_layer = tf.keras.layers.Input(shape=(None,))
        # Crear una capa Lambda que retorna una lista fija de valores
        output_layer = tf.keras.layers.Lambda(lambda x: tf.tile(tf.constant([most_pop]), [tf.shape(x)[0], 1]))(input_layer)
        # Definir el modelo
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        # Esto no vale para nada por que no se entrena, pero se necesita para que funcione la evaluación
        model.compile(loss="mse", optimizer="adam")
        # Parar de medir emisiones        
        if not self.emissions_computed:
            self.emissions_tracker.stop()
        
        return model
        
    def __create_tfdata__(self, dataframe):
        data_x = tf.data.Dataset.from_tensor_slices(dataframe.seq.values)              
        data_y = tf.data.Dataset.from_tensor_slices(dataframe.id_item.values)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        return tf.data.Dataset.zip((data_x, data_y))
    
    def train(self, dev=False, save_model=True):
        # Crear un fichero falso de log para que se pueda generar "best_models.csv"
        fake_log_path = self.MODEL_PATH+"dev/"
        os.makedirs(fake_log_path, exist_ok=True)
        df = pd.DataFrame(zip([0, 1],[0, 0],[0, 0]), columns=["epoch", "val_loss", "val_NDCG@10"])
        df.to_csv(fake_log_path+"log.csv", index=False)
        # Otro para el train final
        if not os.path.exists(self.MODEL_PATH+"log.csv"):
            epoch_time = self.emissions_tracker.final_emissions_data.duration
            df = pd.DataFrame(zip([0],[epoch_time],[0],[0]), columns=["epoch", "e_time", "loss", "r10"])
            df.to_csv(self.MODEL_PATH+"log.csv", index=False)
