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
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + "DATA_byList/" + subset_name + "/df.pickle")
        
        # Renombramos idPOI a itemId y otros
        rev = rev.rename(columns={"idPOI": "itemId", "namePOI": "name", "imageId": "images", "dateReview": "date", "reviewTitle": "title", "LinkReview": "url"})
        rev = rev[['reviewId', 'userId', 'itemId', 'rating', 'date', 'text', 'title', 'url', "name"]]

        # Casting a int de algunas columnas
        rev = rev.astype({'reviewId': 'int64', 'itemId': 'int64', 'rating': 'int64'})
        rev["rating"] = rev["rating"]*10

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        return rev
