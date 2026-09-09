# -*- coding: utf-8 -*-
from src.datasets.text_datasets.TextDataset import TextDataset

import numpy as np
import pandas as pd
from tqdm import tqdm
import mapply


class POIDataset(TextDataset):

    def __init__(self, config, load=None):
        super().__init__(config, load)

    def load_subset(self, subset_name) -> pd.DataFrame:
        """Loads the data for one city, keeping only the relevant columns"""

        # Load reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + "DATA_byList/" + subset_name + "/df.pickle")

        # Rename idPOI to itemId and others
        rev = rev.rename(columns={"idPOI": "itemId", "namePOI": "name", "imageId": "images", "dateReview": "date", "reviewTitle": "title", "LinkReview": "url"})
        rev = rev[['reviewId', 'userId', 'itemId', 'rating', 'date', 'text', 'title', 'url', "name"]]

        # Cast some columns to int
        rev = rev.astype({'reviewId': 'int64', 'itemId': 'int64', 'rating': 'int64'})
        rev["rating"] = rev["rating"]*10

        # Drop empty reviews (NaN text, though some reviews with text=="" remain)
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        return rev
