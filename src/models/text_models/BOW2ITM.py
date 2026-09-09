# -*- coding: utf-8 -*-
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence
from src.Common import print_g

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import normalize
import tensorflow_ranking as tfr
import tensorflow as tf
import pandas as pd
import numpy as np


class BOW2ITM(RSTModel):
    """ Predict, from a review encoded as BOW, the restaurant the review is about """
    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):

        mv = self.CONFIG["model"]["model_version"]
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=(len(self.DATASET.DATA["FEATURES_NAME"]),), name="input_bow"))

        if mv == "0":  # NOTE: this used to be version 3
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        '''
        if mv == "0":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))

        if mv == "1":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))

        if mv == "2":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))

        if mv == "3":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.1))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))

        if mv == "4":
            model.add(tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], name="bow_2_rst", kernel_initializer=tf.keras.initializers.Ones()))
            model.add(tf.keras.layers.Dropout(.5))
            model.add(tf.keras.layers.BatchNormalization(name="bow_2_rst_bn"))
        '''
        activation_function = "sigmoid"

        if activation_function == "sigmoid":
            # loss = tf.keras.losses.BinaryCrossentropy()
            loss = tf.keras.losses.CategoricalCrossentropy()
            # loss = tf.keras.losses.BinaryFocalCrossentropy()

        elif activation_function == "softmax":
            loss = tf.keras.losses.CategoricalCrossentropy()

        model.add(tf.keras.layers.Activation(activation_function, name="output_rst"))
        metrics = ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='top_5'), tf.keras.metrics.TopKCategoricalAccuracy(k=10, name='top_10')]
        metrics = [tfr.keras.metrics.RecallMetric(topn=1, name='r1'), tfr.keras.metrics.RecallMetric(topn=5, name='r5'), tfr.keras.metrics.RecallMetric(topn=10, name='r10'),
                   tfr.keras.metrics.PrecisionMetric(topn=5, name='p5'), tfr.keras.metrics.PrecisionMetric(topn=10, name='p10')]

        model.build(self.CONFIG["model"]["batch_size"])

        optimizer = tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model

    def __create_tfdata__(self, dataframe):

        txt_data = self.DATASET.DATA["BOW_SEQUENCES"][dataframe.bow.values].todense()
        rst_data = dataframe.id_item.values

        data_x = tf.data.Dataset.from_tensor_slices(txt_data)
        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)

        return tf.data.Dataset.zip((data_x, data_y))

    def eval_custom_text(self, text_src):

        text = self.DATASET.prerpocess_text(text_src)
        bow = self.DATASET.DATA["VECTORIZER"].transform([text])
        normed_bow = normalize(bow.todense(), axis=1, norm='l1')
        bow_words = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[np.argwhere(normed_bow[0] > 0)[:, 0]]

        # Get the model that predicts restaurants, along with its relevant weight matrix
        rst_model = self.MODEL
        rst_model_weights = rst_model.get_layer("bow_2_rst").get_weights()[0]

        # Predict 4 restaurants with the corresponding part of the model
        rev_rst_pred = rst_model.predict(normed_bow, verbose=0)
        rev_rst_pred_id = np.apply_along_axis(lambda x: (-x).argsort()[:4], 1, rev_rst_pred)
        recommended_rests = rev_rst_pred_id.flatten()

        print("\n")
        print_g("\'%s\'" % text_src)
        # print("\LEMM: %s" % (text))
        print("\tBOW: %s" % (",".join(bow_words)))

        for idx, rst in enumerate(recommended_rests):
            print(f'\t{idx+1}) {self.DATASET.DATA["TEST"].loc[self.DATASET.DATA["TEST"].id_item == rst]["name"].values[0]}' )

            # Top-20 words most relevant to predicting the selected restaurant
            word_weights = rst_model_weights[:, rst]  # + rst_model_weights_bias
            word_ids = np.argsort(-word_weights)[:20]
            most_relevant_w = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[word_ids]

            # Intersection between the user's words and the restaurant's words
            usr_rst_intr = list(set(np.where(word_weights > 0)[0]).intersection(set(np.argwhere(normed_bow[0] > 0)[:, 0])))
            usr_rst_intr = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[usr_rst_intr]

            print("\t\t▲ %s" % (",".join(most_relevant_w)))
            print("\t\t∩ %s" % (",".join(usr_rst_intr)))

    def explain_test_sample(self, test_real_sample):
        # Get the dataframe row's text and encode it appropriately (since this is a real
        # sample, it has already been encoded).
        bow_encoding = self.DATASET.DATA["BOW_SEQUENCES"][test_real_sample["bow"]].todense()

        # Data for the actual restaurant
        restaurant_name = test_real_sample["name"]

        # Get the relevant weight matrix
        rst_model = self.MODEL
        rst_model_weights = rst_model.get_layer("bow_2_rst").get_weights()[0]

        # Importance of every vocabulary word for this restaurant
        rst_word_weights = rst_model_weights[:, test_real_sample.id_item]

        # Words in the review (the only filtering applied is PoS-based)
        rvw_words = np.array(list(np.argwhere(bow_encoding[0] > 0)[:, 1]))
        rvw_word_names = np.asarray(self.DATASET.DATA["FEATURES_NAME"])[rvw_words]

        print({w_nm: rst_word_weights[w_id] for w_id, w_nm in zip(rvw_words, rvw_word_names)})

        return {"max": rst_model_weights.max(), "min": rst_model_weights.min(), "values": {w_nm: rst_word_weights[w_id] for w_id, w_nm in zip(rvw_words, rvw_word_names)}}
