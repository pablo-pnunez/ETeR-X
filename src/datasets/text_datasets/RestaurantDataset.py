# -*- coding: utf-8 -*-
from src.datasets.text_datasets.TextDataset import TextDataset

import pandas as pd


class RestaurantDataset(TextDataset):

    def __init__(self, config, load=None):
        super().__init__(config, load)

    def load_subset(self, subset_name) -> pd.DataFrame:
        """Loads the data for one city, keeping only the relevant columns"""

        # Load restaurants
        res = pd.read_pickle(self.CONFIG["data_path"] + subset_name + "/restaurants.pkl")

        # Load reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + subset_name + "/reviews.pkl")
        rev = rev[['reviewId', 'userId', 'restaurantId', 'rating', 'date', 'language', 'text', 'title', 'url']]

        # Rename restaurantId to itemId
        rev = rev.rename(columns={"restaurantId": "itemId"})

        # Cast some columns to int
        res = res.astype({'id': 'int64'})
        rev = rev.astype({'reviewId': 'int64', 'itemId': 'int64', 'rating': 'int64'})

        # Merge restaurants and reviews
        rev = rev.merge(res[["id", "name"]], left_on="itemId", right_on="id", how="left")
        rev = rev.drop(columns=["id"])

        # Drop empty reviews (NaN text, though some reviews with text=="" remain)
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        return rev
