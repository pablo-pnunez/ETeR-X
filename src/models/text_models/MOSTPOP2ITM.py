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
        # Track emissions
        if not self.emissions_computed:
            self.emissions_tracker = EmissionsTracker(project_name="Epoch 0", log_level="error", output_dir=self.MODEL_PATH, output_file="emissions_train.csv", tracking_mode="process")
            self.emissions_tracker.start()
        # Number of reviews per user
        item_pop = self.DATASET.DATA["TRAIN_DEV"]["id_item"].value_counts().reset_index().sort_values("index").reset_index(drop=True)["id_item"].values
        # Normalize between 0 and 1 (not strictly needed, but tidy)
        most_pop = np.round((item_pop-min(item_pop))/(max(item_pop)-min(item_pop)),2).tolist()
        # Define the model input
        input_layer = tf.keras.layers.Input(shape=(None,))
        # Create a Lambda layer that returns a fixed list of values
        output_layer = tf.keras.layers.Lambda(lambda x: tf.tile(tf.constant([most_pop]), [tf.shape(x)[0], 1]))(input_layer)
        # Define the model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        # This is meaningless (nothing is trained), but compile() is required for evaluate() to work
        model.compile(loss="mse", optimizer="adam")
        # Stop tracking emissions
        if not self.emissions_computed:
            self.emissions_tracker.stop()

        return model

    def __create_tfdata__(self, dataframe):
        data_x = tf.data.Dataset.from_tensor_slices(dataframe.seq.values)
        data_y = tf.data.Dataset.from_tensor_slices(dataframe.id_item.values)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        return tf.data.Dataset.zip((data_x, data_y))

    def train(self, dev=False, save_model=True):
        # Create a fake log file so that "best_models.csv" can be generated
        fake_log_path = self.MODEL_PATH+"dev/"
        os.makedirs(fake_log_path, exist_ok=True)
        df = pd.DataFrame(zip([0, 1],[0, 0],[0, 0]), columns=["epoch", "val_loss", "val_NDCG@10"])
        df.to_csv(fake_log_path+"log.csv", index=False)
        # Another one for the final training run
        if not os.path.exists(self.MODEL_PATH+"log.csv"):
            epoch_time = self.emissions_tracker.final_emissions_data.duration
            df = pd.DataFrame(zip([0],[epoch_time],[0],[0]), columns=["epoch", "e_time", "loss", "r10"])
            df.to_csv(self.MODEL_PATH+"log.csv", index=False)
