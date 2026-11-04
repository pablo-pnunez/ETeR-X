# -*- coding: utf-8 -*-
from src.Common import print_g
from src.models.KerasModelClass import KerasModelClass
from src.sequences.BaseSequence import BaseSequence

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import tensorflow_ranking as tfr
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MultiLabelBinarizer
from gensim.models import KeyedVectors

class RSTModel(KerasModelClass):
    #TODO change this to ITEM!!
    """ Methods common to every VAL model """
    def __init__(self, config, dataset):
        KerasModelClass.__init__(self, config=config, dataset=dataset)

    def baseline(self, test=False):
        """ Computes popularity so it can act as a baseline, and converts it into a probability """
        if test:
            dataset = self.DATASET.DATA["TRAIN_DEV"]
        else:
            dataset = self.DATASET.DATA["TRAIN_DEV"].loc[self.DATASET.DATA["TRAIN_DEV"]["dev"] == 0]

        y_out = np.zeros((len(dataset), self.DATASET.DATA["N_RST"]))
        y_out[np.expand_dims(list(range(len(y_out))), -1), np.expand_dims(dataset.id_restaurant, -1)] = 1

        popularidad_vec = y_out.sum(axis=0) / sum(y_out.sum(axis=0))
        pred_popularidad = np.row_stack([popularidad_vec]*len(dataset))

        with tf.device('/cpu:0'):

            m1 = tf.keras.metrics.Accuracy()
            m1.update_state(y_true=y_out.argmax(axis=1), y_pred=pred_popularidad.argmax(axis=1))
            # print_g("Popularity ACCURACY: %.4f" % (m1.result().numpy()))
            # m1.reset_states()

            m2 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
            m2.update_state(y_true=y_out, y_pred=pred_popularidad)
            # print_g("Popularity TOP5ACC:  %.4f" % (m2.result().numpy()))
            # m2.reset_states()

            m3 = tf.keras.metrics.TopKCategoricalAccuracy(k=10)
            m3.update_state(y_true=y_out, y_pred=pred_popularidad)
            # print_g("Popularity TOP10ACC:  %.4f" % (m3.result().numpy()))
            # m3.reset_states()

        res = dict(zip(["ACCURACY", "TOP5ACC", "TOP10ACC"], [m1.result().numpy(), m2.result().numpy(), m3.result().numpy()]))
        print_g(res)
        return res

    def explain_test_sample(self, data_row):
        raise NotImplementedError

    def __per_user_ranking_eval__(self, test_data, y_pred, y_true, k=10):
        """ Recomputes NDCG@k/Recall@k by aggregating, per user, all of their test reviews into a
        single ranked list, instead of treating each review as an independent instance
        (which is what Keras' `evaluate()`/`model.evaluate()` above does).

        This replicates the semantics of Cornac's `ranking_eval` (used to evaluate the CF
        baselines -MostPop/EASE^R/BiVAE- in experiments/Baselines.py): a single prediction per user, compared
        against the set of ALL their relevant items in test at once. Without this adjustment, a
        user with several test reviews counts several times in the text models' average
        but only once in the CF baselines', which makes the columns of
        'tab:results_tk1' (Known users) not directly comparable to each other without this
        adjustment.

        For each user:
          - The "score" is the mean of the predictions (y_pred) of their test reviews, as a
            proxy for "a single prediction per user" (our model has no user embedding,
            by design, from which to obtain that single prediction directly).
          - The set of relevant items is the union of the items across all their test
            reviews (usually 1 item per review, though a user can have more than 1 distinct item).

        NOTE: unlike Cornac, this does not exclude from the ranking the items a
        user already reviewed in train (Cornac does so via `gt_neg`); our text models
        do not model which items a user has already seen, so that exclusion has no
        direct equivalent here. Documented as a known limitation of this comparison.

        Args:
            test_data (pd.DataFrame): test rows, aligned 1:1 with y_pred/y_true (same order).
            y_pred (np.ndarray): (n_test_reviews, n_items) matrix of predicted probabilities.
            y_true (np.ndarray): (n_test_reviews, n_items) one-hot matrix of each review's true item.
            k (int): cutoff for NDCG@k/Recall@k.

        Returns:
            pd.DataFrame: one row per user, with NDCG@k/Recall@k aggregated at the user level.
        """
        rows = []
        user_ids = test_data["userId"].values

        for user_id in pd.unique(user_ids):
            idx = np.nonzero(user_ids == user_id)[0]

            score_vec = y_pred[idx].mean(axis=0)
            relevant_vec = y_true[idx].max(axis=0)  # OR-multi-hot of each review's one-hot
            relevant_items = np.nonzero(relevant_vec)[0]

            if len(relevant_items) == 0:
                continue

            ndcg_at_k = ndcg_score([relevant_vec], [score_vec], k=k)

            top_k = np.argsort(-score_vec)[:k]
            hits = len(set(top_k.tolist()) & set(relevant_items.tolist()))
            recall_at_k = hits / len(relevant_items)

            rows.append({
                "userId": user_id,
                f"NDCG@{k}_user": ndcg_at_k,
                f"Recall@{k}_user": recall_at_k,
                "n_test_reviews": len(idx),
                "n_relevant_items": len(relevant_items),
            })

        return pd.DataFrame(rows)

    def __create_tfdata__(self, data):
        raise NotImplementedError

    def get_train_dev_sequences(self, dev):
        all_data = self.DATASET.DATA["TRAIN_DEV"]

        if dev:
            train_data = all_data[all_data["dev"] == 0]
            dev_data = all_data[all_data["dev"] == 1]
            train_gn = self.__create_tfdata__(train_data)
            dev_gn = self.__create_tfdata__(dev_data)
            return train_gn, dev_gn
        else:
            train_dev_gn = self.__create_tfdata__(all_data)
            return train_dev_gn

    def evaluate(self, test=False, user_info=False, verbose=True, out_tag=""):
        if test:
            test_data = self.DATASET.DATA["TEST"]
        else:
            test_data = self.DATASET.DATA["TRAIN_DEV"][self.DATASET.DATA["TRAIN_DEV"]["dev"] == 1]

        test_gn = self.__create_tfdata__(test_data)

        metrics = [
            tfr.keras.metrics.NDCGMetric(topn=1, name="NDCG@1"),
            tfr.keras.metrics.NDCGMetric(topn=10, name="NDCG@10"),
            tfr.keras.metrics.NDCGMetric(topn=50, name="NDCG@50"),
            tfr.keras.metrics.NDCGMetric(topn=100, name="NDCG@100"),# DO NOT use, some datasets have fewer than 100 items
            tfr.keras.metrics.NDCGMetric(name="NDCG@-1"),
            tf.keras.metrics.Precision(name="Precision@-1"),
            tf.keras.metrics.Precision(top_k=1, name="Precision@1"),
            tf.keras.metrics.Precision(top_k=5, name="Precision@5"),
            tf.keras.metrics.Precision(top_k=10, name="Precision@10"),
            tf.keras.metrics.Recall(name="Recall@-1"),
            tf.keras.metrics.Recall(top_k=1, name="Recall@1"),
            tf.keras.metrics.Recall(top_k=5, name="Recall@5"),
            tf.keras.metrics.Recall(top_k=10, name="Recall@10"),
            tf.keras.metrics.Recall(top_k=20, name="Recall@20"),
            tf.keras.metrics.Recall(top_k=50, name="Recall@50"),]

        if verbose: print_g(f"There are {len(test_data)} evaluation examples.")

        self.MODEL.compile(loss=self.MODEL.loss, optimizer=self.MODEL.optimizer, metrics=metrics)
        ret = self.MODEL.evaluate(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0, return_dict=True)

        if user_info:
            y_pred = self.MODEL.predict(test_gn.cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0)
            y_true = np.row_stack([list(f[-1]) for f in test_gn.as_numpy_iterator()])

            sm_all = []
            # metrics = [tfr.keras.metrics.NDCGMetric(name="NDCG@-1")]
            for yp, yt in zip(y_pred, y_true):
                # sm = self.MODEL.evaluate(self.__create_tfdata__(test_data.iloc[idx:idx+1]).cache().batch(self.CONFIG["model"]['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=0)
                sm = ndcg_score([yt], [yp])
                sm_all.append(sm)

            sm_all = pd.DataFrame(sm_all, columns=["NDCG"])
            sm_all = pd.concat([test_data.reset_index(drop=True), sm_all], axis=1)
            train_dev_user_count = self.DATASET.DATA["TRAIN_DEV"].groupby("userId").agg(cold=("userId","count")).reset_index()
            sm_all = sm_all.merge(train_dev_user_count, how="left").fillna(0)
            sm_all.to_csv(f"{self.MODEL_PATH}{'final_' if test else ''}user_eval{out_tag}.csv")

            # Per-user aggregated metrics (comparable to the aggregation Cornac uses
            # for the CF baselines) instead of per-review. See __per_user_ranking_eval__.
            user_agg = self.__per_user_ranking_eval__(test_data.reset_index(drop=True), y_pred, y_true, k=10)
            user_agg.to_csv(f"{self.MODEL_PATH}{'final_' if test else ''}user_agg_eval{out_tag}.csv", index=False)
            if verbose:
                print_g(f"Per-user aggregated (Cornac-style) NDCG@10/Recall@10:\n{user_agg[['NDCG@10_user', 'Recall@10_user']].mean()}")

        # Compute F1 manually
        for r in ["@-1","@1", "@5", "@10"]:
            r_at = ret[f"Recall{r}"]
            p_at = ret[f"Precision{r}"]
            f1_at = 2 * ((r_at * p_at) / (r_at + p_at))
            ret[f"F1{r}"] = f1_at

        ret = pd.DataFrame([ret.values()], columns=ret.keys())
        if verbose: print_g(ret, title=False)

        return ret
