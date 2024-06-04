# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.text_models.RSTModel import RSTModel
from src.sequences.BaseSequence import BaseSequence

from lime.lime_text import LimeTextExplainer

import re
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text 
import tensorflow_ranking as tfr
import keras_nlp

from sklearn.preprocessing import MultiLabelBinarizer


class BERT2ITM(RSTModel):
    """ Predecir, a partir de una review codificada mendiante BERT, el restaurante de la review """

    def __init__(self, config, dataset):
        RSTModel.__init__(self, config=config, dataset=dataset)

    def get_model(self):
        model = self.get_sub_model()
        return model

    def get_sub_model(self):
        mv = self.CONFIG["model"]["model_version"]
        self.MODEL_VERSION = mv

        if mv == "0":
            backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")
            backbone.trainable = True
            inputs = backbone.input
            sequence = backbone(inputs)["sequence_output"]
            # sequence = keras_nlp.layers.TransformerEncoder(num_heads=2, intermediate_dim=256, dropout=0.5)(sequence)

            # Use [CLS] token output to classify
            sequence = tf.keras.layers.Dropout(.8)(sequence)
            outputs = tf.keras.layers.Dense(self.DATASET.DATA["N_ITEMS"], activation="sigmoid", name="out", dtype='float32')(sequence[:, backbone.cls_token_index, :])

            model = tf.keras.Model(inputs, outputs)
            

        metrics = [tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"), tf.keras.metrics.MeanAbsoluteError(name="MAE"),
                    tfr.keras.metrics.RecallMetric(topn=5, name='RC@5'), tfr.keras.metrics.RecallMetric(topn=10, name='RC@10')]

        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(self.CONFIG["model"]["learning_rate"]), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=metrics)

        #print(model.summary())

        return model

    def __create_tfdata__(self, dataframe):
        
        preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")

        # Seleccionar del conjunto donde están todos los textos, los pertenecientes al cojunto "dataframe"
        all_data = pd.read_pickle(self.DATASET.DATASET_PATH+"ALL_DATA")[["reviewId", "text"]]
        text_data = all_data.set_index("reviewId").loc[dataframe.reviewId.values]["text"].values

        seq_data = text_data
        rst_data = dataframe.id_item.values
        
        data_x = tf.data.Dataset.from_tensor_slices(seq_data).map(preprocessor, tf.data.AUTOTUNE)
        data_y = tf.data.Dataset.from_tensor_slices(rst_data)
        data_y = data_y.map(lambda x: tf.one_hot(x, self.DATASET.DATA["N_ITEMS"]), num_parallel_calls=tf.data.AUTOTUNE)
        
        return tf.data.Dataset.zip((data_x, data_y))

    def evaluate_text(self, text):

        print("\n")
        print_g("\'%s\'" % text)

        n_rsts = 4
        # lstm_text = self.DATASET.DATA["TEXT_TOKENIZER"].texts_to_sequences([self.DATASET.prerpocess_text(text)])
        # lstm_text_pad = tf.keras.preprocessing.sequence.pad_sequences(lstm_text, maxlen=self.DATASET.DATA["MAX_LEN_PADDING"])

        # Obtener para la review al completo, el top 3 de restaurantes predichos por el modelo
        # preds_rst = self.MODEL.predict(lstm_text_pad)
        preds_rst = self.MODEL.predict([text], verbose=0).flatten()
        preds_rst_best = np.argsort(-preds_rst)[:n_rsts]

        for rst in preds_rst_best:
            rst_name = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_item == rst]["name"].values[0]
            print(f"\t[{preds_rst[rst]:0.2f}] {rst_name}")

        # Para cada una de las palabras de la review, obtener el restaurante que más importancia le da
        for w in text.split(" "):
            preds_wrd = self.MODEL.predict([w], verbose=0).flatten()
            preds_wrd_best = np.argsort(-preds_wrd)[:1][0]
            rst_word_name = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"].id_item == preds_wrd_best]["name"].values[0]
            print(f"\t ·{w} [{preds_wrd.max():0.2f}] => {rst_word_name}")

    def explain_test_sample(self, test_real_sample):
        '''Extraer con LIME las palabras relevantes del texto que se pasa como parámetro'''

        # Obtenemos la lista de resutanrantes disponibles y creamos el explainer de LIME
        class_names = self.DATASET.DATA["TRAIN_DEV"][["id_item", "name"]].sort_values("id_item").drop_duplicates().reset_index(drop=True)
        explainer = LimeTextExplainer(class_names=class_names.name.values)

        # Función que llama LIME cada vez
        def classifier(d):
            # takes a list of d strings and outputs a (d, k) numpy array with prediction probabilities, where k is the number of classes. For ScikitClassifiers , this is classifier.predict_proba.
            ret = []
            for text in d:
                text = re.sub(r"\s+", " ", text, 0, re.MULTILINE)
                if len(text.strip()) > 0:
                    preds_rst = self.MODEL.predict([text], verbose=0)
                    ret.append(preds_rst.flatten())
                else:
                    ret.append(np.zeros(len(class_names)))

            return np.row_stack(ret)

        # El texto de la review y el id del restaurante
        text_instance = test_real_sample.text
        restaurant_id = test_real_sample.id_item

        exp = explainer.explain_instance(text_instance, classifier, num_samples=500, num_features=10, labels=[restaurant_id])  # ftrs = 10 es el valor por defecto y samples = 5000
        exp = dict(exp.as_list(label=restaurant_id))

        return {"max": max(exp.values()), "min": min(exp.values()), "values": exp}
