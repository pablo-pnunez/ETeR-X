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

        # Lowercase, strip numbers, accents, etc...
        text = self.__preprocess_text_base__(text)

        # Lemmatize if enabled
        text = self.__preprocess_text_lemma__(text)

        return text

    def __preprocess_text_base__(self, text):

        # Lowercase
        text = text.lower()

        # Remove formatting characters (\n \t ...)
        rgx_b = r'(\\.)+'
        text = re.sub(rgx_b, ' ', text).strip()

        # Replace punctuation with spaces
        rgx_a = r'\s*[^\w\s]+\s*'
        text = re.sub(rgx_a, ' ', text).strip()

        # Remove accents? Does not strip the tilde on "ñ"
        if self.CONFIG["remove_accents"]:
            rgx_c = r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+"
            text = normalize_uni('NFC', re.sub(rgx_c, r"\1", normalize_uni("NFD", text), 0, re.I))

        # Remove numbers?
        if self.CONFIG["remove_numbers"]:
            rgx_d = r"\s*\d+\s*"
            text = re.sub(rgx_d, ' ', text).strip()

        return text

    def __preprocess_text_lemma__(self, text):

        # Tagging & lemmatization
        if self.CONFIG["lemmatization"]:
            text = " ".join([e.lemma_ for e in self.NLP(text)])  # some words end up capitalized
            # Preprocess again, since lemmatization can reintroduce accents and capitals
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

        # Load the reviews
        rev = self.load_subset(subset_name)

        # Preprocess the stopwords
        self.STOPWORDS = self.prerpocess_text(" ".join(self.STOPWORDS)).split(" ")

        # Preprocess the texts
        # mapply.init(n_workers=-1, chunk_size=100, max_chunks_per_worker=8, progressbar=True)

        # Keep the original text (with casing/punctuation intact) for PoS tagging: this is
        # necessary to correctly distinguish proper nouns (PROPN) from common nouns
        # (NOUN), a signal spaCy loses if tagging is done on already-lowercased text.
        rev["text_raw"] = rev["text"]

        # First preprocess the text, removing basic elements (lowercase, strip numbers)
        # rev["text"] = rev["text"].mapply(self.__preprocess_text_base__)
        rev["text"] = rev["text"].apply(self.__preprocess_text_base__)

        # Then lemmatize if needed, and use the opportunity to run PoS tagging
        # FIXME: Language-extraction code duplicated
        pos_list = []
        if self.CONFIG["lemmatization"]:
            out_list = []
            lang_list = []
            for doc in tqdm(self.NLP.pipe(rev["text_raw"].tolist(), n_process=10), total=len(rev), desc="Lemmatization + PoS"):
                # The lemma needs preprocessing too, it can have accents and capitals
                # Build word, lemma, pos triples
                trias = [(str(f), self.__preprocess_text_base__(f.lemma_), f.pos_) for f in doc]
                pos_list.extend(trias)
                # Add the lemmatized text to a list
                lema = " ".join(map(lambda x: x[1], trias))
                out_list.append(lema)
                # Extract the review's language
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
            # Extract the review's language without lemmatization
            lang_list = []
            for doc in tqdm(self.NLP.pipe(rev["text"].tolist()), total=len(rev), desc="Language extraction"):
                lang_list.append(doc._.language["language"])
            rev["lang"] = lang_list

        # Drop reviews not in the target language (save the rest to excel)
        rev[rev["lang"] != self.CONFIG["language"]][["text", "lang"]].to_excel(self.DATASET_PATH+"other_languages.xlsx")
        rev = rev[rev["lang"] == self.CONFIG["language"]]

        # Get the word count of the review and title
        rev["n_words_text"] = rev["text"].apply(lambda x: 0 if len(x) == 0 else len(x.split(" ")))

        # Drop reviews with zero-length text or title
        rev = rev.loc[(rev["n_words_text"] > 0)]

        # Get the character count of the reviews and drop those with more than 2000
        rev["n_char_text"] = rev["text"].apply(lambda x: len(x))
        rev = rev.loc[rev["n_char_text"] <= 2000]

        return rev, pos_list

    def load_subset(self, subset_name) -> pd.DataFrame:
        """
        Loads the data for a subset (city, category)
        Arguments:
            subset_name: the name of the subset (city, category)
        Returns:
            Dictionary with the loaded or generated data (if it didn't already exist)
        """
        # raise NotImplemented
        return []

    def get_data(self, load=["TEXT_SEQUENCES", "BOW_SEQUENCES", "TRAIN_DEV", "TEST", "WORD_INDEX", "VOCAB_SIZE", "MAX_LEN_PADDING", "TEXT_TOKENIZER", "VECTORIZER", "FEATURES_NAME", "N_ITEMS"]):

        # Load the data
        dict_data = self.get_dict_data(self.DATASET_PATH, load)

        # If it already exists, return it
        if dict_data is not False and len(dict_data) == len(load):
            return dict_data
        # If it doesn't exist, create it
        else:

            # Load the reviews
            all_data, pos_data = self.__load_subset__(subset_name=self.CONFIG["subset"])

            # Items with X or more reviews
            r_mth_x = all_data.groupby("itemId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_rst"] else 1).reset_index()
            r_mth_x = r_mth_x.loc[r_mth_x[0] == 1]["itemId"].values
            print_g("%d items with %d or more reviews." % (len(r_mth_x), self.CONFIG["min_reviews_rst"]))
            all_data = all_data.loc[all_data["itemId"].isin(r_mth_x)]

            # Users with X or more reviews
            u_mth_x = all_data.groupby("userId").apply(lambda x: 0 if len(x) < self.CONFIG["min_reviews_usr"] else 1).reset_index()
            u_mth_x = u_mth_x.loc[u_mth_x[0] == 1]["userId"].values
            all_data = all_data.loc[all_data["userId"].isin(u_mth_x)]

            # Get the resulting set's stats after filtering (for the paper)
            print("· Number of resulting examples: %d" % len(all_data))
            print("· Number of items: %d" % len(all_data.itemId.unique()))
            print("· Number of users: %d" % len(all_data.userId.unique()))
            print("· Average items per user: %f" % all_data.groupby("userId").apply(lambda x: len(x.itemId.unique())).mean())
            os.makedirs("data/raw/", exist_ok=True)
            all_data[["reviewId", "userId", "itemId", "rating", "text"]].to_pickle("data/raw/%s.pkl" % self.CONFIG["subset"])

            # Create item ids (for the ONE-HOT encoding)
            rst_newid = pd.DataFrame(zip(r_mth_x, range(len(r_mth_x))), columns=["itemId", "id_item"])
            all_data = all_data.merge(rst_newid, on="itemId")
            all_data = all_data.drop(columns=["itemId"])

            # Shuffle the reviews
            all_data = all_data.sample(frac=1, random_state=self.CONFIG["seed"]).reset_index(drop=True)

            # Build the BOW vectors
            if self.CONFIG["remove_stopwords"] == 0:  # Most frequent only
                vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 1:  # Most frequent + manual stopwords
                print("Update the code for other languages")
                return NotImplemented
                vectorizer = CountVectorizer(stop_words=self.STOPWORDS, min_df=self.CONFIG["min_df"], max_features=self.CONFIG["bow_pct_words"], binary=self.CONFIG["presencia"])
            elif self.CONFIG["remove_stopwords"] == 2:  # Most frequent + automatic stopwords
                if self.CONFIG["lemmatization"]:
                    # Run a CountVectorizer over every word to get each one's frequency
                    vectorizer = CountVectorizer(stop_words=None, min_df=self.CONFIG["min_df"], max_features=None, binary=self.CONFIG["presencia"])
                    bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])
                    word_freq = np.asarray(bow.sum(axis=0))[0]
                    features = vectorizer.get_feature_names_out()

                    # Each lemma's most common PoS tag was already computed in __load_subset__ (pos_data),
                    # tagging the ORIGINAL text (with casing/punctuation intact) so as not to
                    # lose the signal that distinguishes proper nouns (PROPN) from common nouns
                    # (NOUN). Reusing it here also avoids re-tagging the whole corpus a second time.
                    lemma_pos_counts = pos_data.groupby(["lemma", "pos"]).agg(times=("times", "sum")).reset_index()
                    word_pos_map = lemma_pos_counts.loc[lemma_pos_counts.groupby("lemma")["times"].idxmax()].set_index("lemma")["pos"]
                    word_pos = pd.Series(features).map(word_pos_map).fillna("X").values

                    # Store everything in a DF to look up the X most frequent words meeting the requirements
                    word_data = pd.DataFrame(zip(features, word_freq, word_pos), columns=["feature", "freq", "pos"]).sort_values("freq", ascending=False).reset_index(drop=True)
                    word_data.to_excel(self.DATASET_PATH+"all_features.xlsx")

                    selected_words = word_data.loc[word_data.pos.isin(["ADJ", "NOUN"])]
                    num_selected_words = int(len(selected_words)*(self.CONFIG["bow_pct_words"]/100))
                    selected_words = selected_words.iloc[:num_selected_words].reset_index(drop=True)
                    # Everything not selected is treated as a stopword
                    stop_words = word_data.loc[~word_data.feature.isin(selected_words.feature)].feature.tolist()
                    vectorizer = CountVectorizer(stop_words=stop_words, min_df=self.CONFIG["min_df"], max_features=num_selected_words, binary=self.CONFIG["presencia"])
                else:
                    print_e("Automatic word selection requires lemmatization.")
                    exit()

            bow = vectorizer.fit_transform(all_data[self.CONFIG["text_column"]])

            # Each word corresponds to a column among the <self.CONFIG["bow_pct_words"]>
            features_name = vectorizer.get_feature_names_out()
            np.savetxt(self.DATASET_PATH+"features.csv", features_name, fmt="%s", delimiter=",")

            # Normalize each review's vector
            # normed_bow = normalize(bow.todense(), axis=1, norm='l1')
            normed_bow = normalize(bow, axis=1, norm='l1')

            # Add the BOW to the data
            # all_data["bow"] = normed_bow.tolist()

            # Moved to a separate file
            # all_data["bow"] = list(map(csc_matrix, normed_bow))
            all_data["bow"] = range(normed_bow.shape[0])

            # Tokenize the words (map each word to an index [WORD_INDEX])
            if self.CONFIG["n_max_words"] == 0:
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer()
            elif self.CONFIG["n_max_words"] > 0:
                n_max_words = self.CONFIG["n_max_words"]
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)
            else:
                # Find the maximum number of words based on each one's frequency
                tokenizer_tmp = tf.keras.preprocessing.text.Tokenizer()
                tokenizer_tmp.fit_on_texts(all_data[self.CONFIG["text_column"]])
                token_freq = pd.DataFrame(zip(tokenizer_tmp.word_counts.keys(), tokenizer_tmp.word_counts.values()), columns=["token", "freq"]).sort_values("freq", ascending=False).reset_index(drop=True)
                token_freq = token_freq[token_freq.freq >= abs(self.CONFIG["n_max_words"])]
                n_max_words = len(token_freq)
                tokenizer_txt = tf.keras.preprocessing.text.Tokenizer(num_words=n_max_words)

            tokenizer_txt.fit_on_texts(all_data[self.CONFIG["text_column"]])
            word_index = tokenizer_txt.word_index

            # If all of them are used
            if self.CONFIG["n_max_words"] == 0:
                print_g("%d words involved and we use them all." % len(word_index))
            # Keep only the "n_max_words" most frequent words
            else:
                word_counts = tokenizer_txt.word_counts
                word_counts = {k: v for k, v in sorted(word_counts.items(), key=lambda item: item[1], reverse=True)}
                mas_frecuentes = list(word_counts.keys())[0:n_max_words]
                print_g("%d words involved, but we keep only the %d most frequent." % (len(word_index), n_max_words))
                word_index = {x: word_index[x] for x in mas_frecuentes}

            # Turn sentences into sequences of numbers per [WORD_INDEX] and add them to the set
            text_sequences = tokenizer_txt.texts_to_sequences(all_data[self.CONFIG["text_column"]])
            all_data["seq"] = text_sequences

            # Train/test split making sure Train contains every item.
            def data_split(rst_rvws):
                for_dev_tst = int(len(rst_rvws) * (self.CONFIG["test_dev_split"]*2))//2
                if for_dev_tst > 0:
                    rst_rvws.iloc[-for_dev_tst:, rst_rvws.columns.get_loc("test")] = 1  # Last x go to test
                    rst_rvws.iloc[-for_dev_tst*2:-for_dev_tst, rst_rvws.columns.get_loc("dev")] = 1  # Second-to-last x go to dev
                return rst_rvws

            all_data["dev"] = 0
            all_data["test"] = 0
            all_data = all_data.groupby("id_item").apply(data_split)

            # Truncate the padding?
            max_len_padding = None
            if self.CONFIG["truncate_padding"]:
                seq_lens = all_data.loc[(all_data["dev"] == 0) & (all_data["test"] == 0)]["seq"].apply(lambda x: len(x)).values
                max_len_padding = int(seq_lens.mean() + seq_lens.std() * 2)

            # Add to the set with the padding
            seq_w_pad = tf.keras.preprocessing.sequence.pad_sequences(all_data["seq"].values, maxlen=max_len_padding).astype(np.int32)

            # Add an id to be able to store the sequences in a numpy array
            all_data["seq"] = range(len(seq_w_pad))
            # all_data["seq"] = seq_w_pad.tolist()
            max_len_padding = seq_w_pad.shape[1]

            # Cast to save space
            all_data = all_data.astype({'dev': 'int8', 'test': 'int8', 'rating': 'int8', 'bow': 'int32', 'seq': 'int32', 'id_item': 'int32', })
            all_data = all_data.astype({'n_words_text': 'int32', 'n_char_text': 'int32'})

            # The text itself isn't needed here, it's kept in a separate DataFrame to avoid bloating RAM
            model_data = all_data[["reviewId", "userId", "id_item", "name", "bow", "seq", "dev", "test"]]

            # Split into the final sets
            train_dev = model_data.loc[model_data["test"] == 0].drop(columns=["test"])
            test = model_data.loc[model_data["test"] == 1].drop(columns=["dev", "test"])

            # Store the pickles
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
