# -*- coding: utf-8 -*-
from src.datasets.DatasetClass import DatasetClass
from src.Common import to_pickle, print_g, print_e

import os
import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from nltk.corpus import stopwords
from spacy.language import Language
from sklearn.preprocessing import normalize
from spacy_langdetect import LanguageDetector
from unicodedata import normalize as normalize_uni
from sklearn.feature_extraction.text import CountVectorizer


def get_lang_detector(nlp, name):
    return LanguageDetector()  # We use the seed 42


class TextDataset(DatasetClass):

    def __init__(self, config, load):
        nltk.download('stopwords', quiet=True)

        if config["language"] == "es":
            import es_core_news_sm as spacy_model  # python -m spacy download es_core_news_sm
        elif config["language"] == "en":
            import en_core_web_sm as spacy_model
        elif config["language"] == "fr":
            import fr_core_news_sm as spacy_model
        else:
            raise Exception

        # self.NLP = spacy_model.load(disable=["parser", "ner", "attribute_ruler"])
        # self.NLP = spacy_model.load(disable=["parser", "ner"])
        self.NLP = spacy_model.load()
        Language.factory("language_detector", func=get_lang_detector)
        self.NLP.add_pipe('language_detector', last=True)
        self.STOPWORDS = self.__get_stopwords__(lang=config["language"])

        DatasetClass.__init__(self, config=config, load=load)

    def prerpocess_text(self, text):

        # A minusculas, eliminar números, acentos etc...
        text = self.__preprocess_text_base__(text)

        # Hacer lemmatización si está activado
        text = self.__preprocess_text_lemma__(text)

        return text

    def __preprocess_text_base__(self, text):

        # A minusculas
        text = text.lower()

        # Eliminar formatos (/n /t ...)
        rgx_b = r'(\\.)+'
        text = re.sub(rgx_b, ' ', text).strip()

        # Cambiar signos de puntuación por espacios
        rgx_a = r'\s*[^\w\s]+\s*'
        text = re.sub(rgx_a, ' ', text).strip()

        # Eliminar accentos? No quita eñes
        if self.CONFIG["remove_accents"]:
            rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
            text = normalize_uni('NFC', re.sub(rgx_c, r"\1", normalize_uni("NFD", text), 0, re.I))

        # Eliminar números?
        if self.CONFIG["remove_numbers"]:
            rgx_d = r"\s*\d+\s*"
            text = re.sub(rgx_d, ' ', text).strip()

        return text

    def __preprocess_text_lemma__(self, text):

        # Tagging & Lemmatización
        if self.CONFIG["lemmatization"]:
            text = " ".join([e.lemma_ for e in self.NLP(text)])  # alguna palabra la pone mayúscula
            # Preprocesar de nuevo, la lemmatización añade acentos y mayúsculas
            text = self.__preprocess_text_base__(text=text)

        return text

    def __get_stopwords__(self, lang="es"):
        # nltk.download('stopwords')
        ret_stgrs = []

        if lang == "es":
            ret_stgrs = stopwords.words('spanish')
        elif lang == "en":
            ret_stgrs = stopwords.words("english")
        elif lang == "fr":
            ret_stgrs = stopwords.words("french")
        else:
            raise NotImplemented

        return ret_stgrs

    def __load_subset__(self, subset_name):

        # Cargamos las reviews
        rev = self.load_subset(subset_name)

        # Preprocesar las StopWords
        self.STOPWORDS = self.prerpocess_text(" ".join(self.STOPWORDS)).split(" ")

        # Preprocesar los textos
        # mapply.init(n_workers=-1, chunk_size=100, max_chunks_per_worker=8, progressbar=True)

        # Primero preprocesamos el texto eliminando elementos básicos (a minusculas, quitar números)
        # rev["text"] = rev["text"].mapply(self.__preprocess_text_base__)
        rev["text"] = rev["text"].apply(self.__preprocess_text_base__)

        # Luego hacemos la lemmatización si es necesario y aprovechamos para hacer el PoS
        # FIXME: Código de extraer lenguaje repetido
        pos_list = []
        if self.CONFIG["lemmatization"]:
            out_list = []
            lang_list = []
            for doc in tqdm(self.NLP.pipe(rev["text"].tolist(), n_process=10), total=len(rev), desc="Lemmatization + PoS"):
                # Hay que preprocesar la lemmatización, puede tener tildes y mayúsculas
                # Creamos triplas palabra, lemma, pos
                trias = [(str(f), self.__preprocess_text_base__(f.lemma_), f.pos_) for f in doc]
                pos_list.extend(trias)
                # Añadimos el texto lematizado a una lista
                lema = " ".join(map(lambda x: x[1], trias))
                out_list.append(lema)
                # Extraer el idioma de la reseña
                lang_list.append(doc._.language["language"])

            pos_list = pd.DataFrame(pos_list, columns=["term", "lemma", "pos"])
            pos_list = pos_list.groupby(["term", "pos", "lemma"]).agg(times=("lemma", "count")).reset_index()
            try:
                pos_list.to_excel(self.DATASET_PATH+"term_lemma_pos.xlsx")
            except Exception as e:
                print(e)
            
            rev["text_source"] = rev["text"]
            rev["text"] = out_list
            rev["lang"] = lang_list
            del out_list
        else:
            # Extraer el idioma de la reseña sin lemmatización
            lang_list = []
            for doc in tqdm(self.NLP.pipe(rev["text"].tolist()), total=len(rev), desc="Language extraction"):
                lang_list.append(doc._.language["language"])
            rev["lang"] = lang_list

        # Eliminar reviews que no sean del mismo idioma (guardar el resto en excel)
        rev[rev["lang"] != self.CONFIG["language"]][["text", "lang"]].to_excel(self.DATASET_PATH+"other_languages.xlsx")
        rev = rev[rev["lang"] == self.CONFIG["language"]]

        # Obtener número de palabras de las reviews y del título
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Eliminar reviews de tamaño 0 en texto y titulo
        rev = rev.loc[(rev["n_words_text"] > 0)]

        # Obtener número de caracteres de las reviews y eliminar aquellas con más de 2000
        rev["n_char_text"] = rev["text"].apply(lambda x: len(x))
        rev = rev.loc[rev["n_char_text"] <= 2000]

        return rev, pos_list

    def load_subset(self, subset_name) -> pd.DataFrame:
        """
        Carga los datos de un subconjunto (ciudad, categoría)
        Arguments:
            subset_name: el nombre del subconjunto (ciudad, categoría)
        Returns:
            Diccionario con los datos cargados o generados (si no existían)
        """
        # raise NotImplemented
        return []

    def get_data(self, load=["TEXT_SEQUENCES", "BOW_SEQUENCES", "TRAIN_DEV", "TEST", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "TEXT_TOKENIZER", "VECTORIZER", "FEATURES_NAME", "N_ITEMS"]):

        # Cargar los datos
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # Si ya existen, retornar
        if dict_data is not False and len(dict_data) == len(load):
            return dict_data
        # Si no existe, crear
        else:

            # Cargar las reviews
            all_data, pos_data = self.__load_subset__(subset_name=self.CONFIG["subset"])

            # Restaurantes con X o más reseñas
            r_mth_x = all_data.groupby("itemId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_rst"] else 1).reset_index()
            r_mth_x = r_mth_x.loc[r_mth_x[0] == 1]["itemId"].values
            print_g("%d items with %d or more reviews." % (len(r_mth_x), self.CONFIG["min_reviews_rst"]))
            all_data = all_data.loc[all_data["itemId"].isin(r_mth_x)]

            # Usuarios con X o más reseñas
            u_mth_x = all_data.groupby("userId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_usr"] else 1).reset_index()
            u_mth_x = u_mth_x.loc[u_mth_x[0] == 1]["userId"].values
            all_data = all_data.loc[all_data["userId"].isin(u_mth_x)]

            # Obtener los datos del conjunto tras el filtrado (para el paper)
            print("· Nº de ejemplos resultantes: %d" % len(all_data))
            print("· Nº de items: %d" % len(all_data.itemId.unique()))
            print("· Nº de usuarios: %d" % len(all_data.userId.unique()))
            print("· Nº items medio por usuario: %f" % all_data.groupby("userId").apply(lambda x: len(x.itemId.unique())).mean())
            os.makedirs("data/raw/", exist_ok=True)
            all_data[["reviewId", "userId", "itemId", "rating", "text"]].to_pickle("data/raw/%s.pkl" % self.CONFIG["subset"])

            # Crear id de items (para el ONE-HOT)
            rst_newid = pd.DataFrame(zip(r_mth_x, range(len(r_mth_x))), columns=["itemId", "id_item"])
            all_data = all_data.merge(rst_newid, on="itemId")
            all_data = all_data.drop(columns=["itemId"])

            # Mezclar las reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Crear vectores del BOW
            if self.CONFIG["remove_stopwords"] == 0:  # Solo más frecuentes
                vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 1:  # Más frecuentes + stopwords manual
                print("Actualizar código para otros idiomas")
                return NotImplemented
                vectorizer = CountVectorizer(stop_words=self.STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 2:  # Más frecuentes + stopwords automático
                if self.CONFIG["lemmatization"]:
                    # Se hace un countvectorizer con todas las palabras para obtener la frecuencia de cada una
                    vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=None, binary=self.CONFIG["presencia"])
                    bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])
                    word_freq = np.asarray(bow.sum(axis=0))[0]

                    # Hay que obtener el POS (part of speech) de cada palabra en su contexto (si hay varios, el más habitual)
                    pos_values = np.array(["ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", "SPACE"])

                    # · Obtenemos los conjuntos relevantes
                    RVW_CP = all_data[[self.CONFIG["text_column"]]].copy()
                    # RVW_CP[self.CONFIG["text_column"]] = RVW_CP[self.CONFIG["text_column"]].apply(lambda x: x.split())                
                    features = vectorizer.get_feature_names_out()
                    
                    # · Luego se crear varias estructuras de datos para almacenar los valores de POS
                    features_df = pd.DataFrame(zip(range(len(features)), features), columns=["id_feature", "feature"])
                    features_df = features_df.set_index("feature")
                    pos_df = pd.DataFrame(zip(range(len(pos_values)), pos_values), columns=["id_pos", "pos"])
                    pos_df = pos_df.set_index("pos")
                    mtrx = np.zeros((len(features), len(pos_values)), dtype=int)

                    # · Hacemos 32 batches para evitar sobrecarga de RAM
                    batches = np.array_split(RVW_CP, 32)

                    # · Para cada batch, obtener POS de sus palabras y almacenar valores en mtrx
                    for idx, b in tqdm(enumerate(batches), desc="Features POS"):
                        POS = list(self.NLP.pipe(b.text, n_process=8))  # 8 es lo mejor

                        all_words = np.concatenate(list(map(lambda x: [(str(w), w.pos_) for w in x], POS)))
                        all_df = pd.DataFrame(all_words, columns=["feature", "pos"])
                        all_df = all_df.loc[all_df.feature.isin(features)].reset_index(drop=True)
                        all_df = all_df.merge(features_df, on="feature").merge(pos_df, on="pos")
                        all_df = all_df.groupby(["id_feature", "id_pos"]).apply(len).reset_index()

                        mtrx[all_df.id_feature, all_df.id_pos] += all_df[0]

                        del all_df, b, POS

                    # · Obtener los POS de las features (más común)
                    word_pos = pos_values[np.apply_along_axis(np.argmax, 1, mtrx)]

                    # Se alamacena todo en un DF para buscar X palabras más frecuentes que cumplan las exigencias
                    word_data = pd.DataFrame(zip(features, word_freq, word_pos), columns=["feature", "freq", "pos"]).sort_values("freq", ascending=False).reset_index(drop=True)
                    word_data.to_excel(self.DATASET_PATH+"all_features.xlsx")

                    selected_words = word_data.loc[word_data.pos.isin(["ADJ", "NOUN"])]
                    num_selected_words = int(len(selected_words)*(self.CONFIG["bow_pct_words"]/100))
                    selected_words = selected_words.iloc[:num_selected_words].reset_index(drop=True)
                    # Todas las que no sean seleccionadas, se consideran stopwords
                    stop_words = word_data.loc[~word_data.feature.isin(selected_words.feature)].feature.tolist()
                    vectorizer = CountVectorizer(stop_words=stop_words, min_df=self.CONFIG["min_df"], max_features=num_selected_words, binary=self.CONFIG["presencia"])
                else:
                    print_e("La selección automática de palabras requiere de lemmatización.")
                    exit()

            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # Cada palabra corresponde con cada columna de las <self.CONFIG["bow_pct_words"]>
            features_name = vectorizer.get_feature_names_out()
            np.savetxt(self.DATASET_PATH+"features.csv", features_name, fmt="%s", delimiter=",")

            # Normalizar vector de cada review
            # normed_bow = normalize(bow.todense(), axis=1, norm='l1')
            normed_bow = normalize(bow, axis=1, norm='l1')

            # Incroporar BOW en los datos
            # all_data["bow"] = normed_bow.tolist()
            
            # Se mueve a fichero separado
            # all_data["bow"] = list(map(csc_matrix, normed_bow))
            all_data["bow"] = range(normed_bow.shape[0])

            # Tokenizar las palabras (Asociar cada palabra a un índice [WORD_INDEX])
            if self.CONFIG["n_max_words"] == 0:
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer()
            elif self.CONFIG["n_max_words"] > 0:
                n_max_words = self.CONFIG["n_max_words"]
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)
            else:
                # Se busca el número máximo de palabras en función de la frecuencia de cada una
                tokenizer_tmp = tf.keras.preprocessing.text.Tokenizer()
                tokenizer_tmp.fit_on_texts(all_data[self.CONFIG["text_column"]])
                token_freq = pd.DataFrame(zip(tokenizer_tmp.word_counts.keys(), tokenizer_tmp.word_counts.values()), columns=["token", "freq"]).sort_values("freq", ascending=False).reset_index(drop=True)
                token_freq = token_freq[token_freq.freq >= abs(self.CONFIG["n_max_words"])]
                n_max_words = len(token_freq)
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)

            tokenizer_txt.fit_on_texts(all_data[self.CONFIG["text_column"]])
            word_index = tokenizer_txt.word_index

            # Si se utilizan todas
            if self.CONFIG["n_max_words"] == 0:
                print_g("Intervienen %d palabras y utilizamos todas." % len(word_index))
            # Quedarse con las "n_max_words" palabras más frecuentes
            else:
                word_counts = tokenizer_txt.word_counts
                word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
                mas_frecuentes = list(word_counts.keys())[0:n_max_words]
                print_g("Intervienen %d palabras, pero nos quedamos con las %d más frecuentes." % (len(word_index), n_max_words))
                word_index = {x: word_index[x] for x in mas_frecuentes}

            # Transformar frases a secuencias de números según [WORD_INDEX] y añadir al set
            text_sequences = tokenizer_txt.texts_to_sequences(all_data[self.CONFIG["text_column"]])
            all_data["seq"] = text_sequences

            # Train Test split asegurandose de que en Train están todos los items.
            def data_split(rst_rvws):
                for_dev_tst = int(len(rst_rvws) * (self.CONFIG["test_dev_split"]*2))//2
                if for_dev_tst > 0:
                    rst_rvws.iloc[-for_dev_tst:, rst_rvws.columns.get_loc("test")] = 1  # Últimas x para test
                    rst_rvws.iloc[-for_dev_tst*2:-for_dev_tst, rst_rvws.columns.get_loc("dev")] = 1  # Penúltimas x para dev
                return rst_rvws

            all_data["dev"] = 0
            all_data["test"] = 0
            all_data = all_data.groupby("id_item").apply(data_split)

            # Truncar el padding?
            max_len_padding = None
            if self.CONFIG["truncate_padding"]:
                seq_lens = all_data.loc[(all_data["dev"] == 0) & (all_data["test"] == 0)]["seq"].apply(lambda x: len(x)).values
                max_len_padding = int(seq_lens.mean() + seq_lens.std() * 2)

            # Añadir al set con el padding
            seq_w_pad = tf.keras.preprocessing.sequence.pad_sequences(all_data["seq"].values, maxlen=max_len_padding).astype(np.int32)
            
            # Añadir un id para poder almacenar las secuecias en un numpy
            all_data["seq"] = range(len(seq_w_pad))
            # all_data["seq"] = seq_w_pad.tolist()
            max_len_padding = seq_w_pad.shape[1]

            # Hacer castings para ahorrar espacio
            all_data = all_data.astype({'dev': 'int8', 'test': 'int8', 'rating': 'int8', 'bow': 'int32', 'seq': 'int32', 'id_item': 'int32', })
            all_data = all_data.astype({'n_words_text': 'int32', 'n_char_text': 'int32'})

            # El texto en si no es necesario, se pone en un pandas aparte para no sobrecargar la RAM
            model_data = all_data[["reviewId", "userId", "id_item", "name", "bow", "seq", "dev", "test"]]

            # Separar los conjuntos finales
            train_dev = model_data.loc[model_data["test"] == 0].drop(columns=["test"])
            test = model_data.loc[model_data["test"] == 1].drop(columns=["dev", "test"])

            # Almacenar pickles
            to_pickle(self.DATASET_PATH, "VECTORIZER", vectorizer)
            to_pickle(self.DATASET_PATH, "FEATURES_NAME", features_name)
            to_pickle(self.DATASET_PATH, "N_ITEMS", len(all_data["id_item"].unique()))
            to_pickle(self.DATASET_PATH, "BOW_SEQUENCES", normed_bow)

            to_pickle(self.DATASET_PATH, "WORD_INDEX", word_index)
            to_pickle(self.DATASET_PATH, "VOCAB_SIZE", len(word_index) + 1)
            to_pickle(self.DATASET_PATH, "MAX_LEN_PADDING", max_len_padding)
            to_pickle(self.DATASET_PATH, "TEXT_TOKENIZER", tokenizer_txt)
            to_pickle(self.DATASET_PATH, "TEXT_SEQUENCES", seq_w_pad)

            to_pickle(self.DATASET_PATH, "ALL_DATA", all_data)
            to_pickle(self.DATASET_PATH, "TRAIN_DEV", train_dev)
            to_pickle(self.DATASET_PATH, "TEST", test)

            return self.get_dict_data(self.DATASET_PATH, load)

    def get_data_stats(self):
        all_data = pd.read_pickle(self.DATASET_PATH + "ALL_DATA")

        n_revs = len(all_data["reviewId"].unique())
        n_items = len(all_data["id_item"].unique())
        n_usrs = len(all_data["userId"].unique())

        revs_per_itm = all_data.groupby("id_item").apply(lambda x: len(x.reviewId.unique()))
        avg_revs_itm = revs_per_itm.mean()
        pctg_total_revs_itm_pop = revs_per_itm.sort_values().iloc[-1] / n_revs

        revs_per_usr = all_data.groupby("userId").apply(lambda x: len(x.reviewId.unique()))
        avg_revs_usr = revs_per_usr.mean()
        pctg_total_revs_usr_pop = revs_per_usr.sort_values().iloc[-1] / n_revs

        avg_rating = all_data["rating"].mean() / 10
        std_rating = all_data["rating"].std() / 10

        print("\n".join(map(str, [self.CONFIG["subset"], n_revs, n_items, avg_revs_itm, pctg_total_revs_itm_pop, n_usrs, avg_revs_usr, pctg_total_revs_usr_pop, avg_rating, std_rating])))
