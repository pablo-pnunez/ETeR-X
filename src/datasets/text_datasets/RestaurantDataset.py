# -*- coding: utf-8 -*-
from src.datasets.text_datasets.TextDataset import TextDataset

import pandas as pd


class RestaurantDataset(TextDataset):

    def __init__(self, config, load=None):
        super().__init__(config, load)

    def load_subset(self, subset_name) -> pd.DataFrame:
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar restaurantes
        res = pd.read_pickle(self.CONFIG["data_path"] + subset_name + "/restaurants.pkl")

        # Cargar reviews
        rev = pd.read_pickle(self.CONFIG["data_path"] + subset_name + "/reviews.pkl")
        rev = rev[['reviewId', 'userId', 'restaurantId', 'rating', 'date', 'language', 'text', 'title', 'url']]

        # Renombramos restaurantId a itemId
        rev = rev.rename(columns={"restaurantId": "itemId"})

        # Casting a int de algunas columnas
        res = res.astype({'id': 'int64'})
        rev = rev.astype({'reviewId': 'int64', 'itemId': 'int64', 'rating': 'int64'})

        # Concatenar restaurantes y reviews
        rev = rev.merge(res[["id", "name"]], left_on="itemId", right_on="id", how="left")
        rev = rev.drop(columns=["id"])

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        return rev
