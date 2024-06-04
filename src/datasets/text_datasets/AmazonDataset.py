# -*- coding: utf-8 -*-
from src.datasets.text_datasets.TextDataset import TextDataset

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pyquery import PyQuery as pq
from tqdm import tqdm
import pandas as pd
import requests
import gzip
import json


class AmazonDataset(TextDataset):

    def __init__(self, config, load=None):
        super().__init__(config, load)

    def __dowload_item_name__(self, item):
        baseUrl = "http://www.amazon.com/dp/"

        headersList = { 
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'accept-encoding': 'gzip, deflate, br',
            'accept-language': 'es,en;q=0.9,ca;q=0.8',
            'cache-control': 'no-cache',
            'device-memory': '8',
            'dnt': '1',
            'downlink': '10',
            'dpr': '1',
            'ect': '4g',
            'pragma': 'no-cache',
            'rtt': '50',
            'sec-ch-device-memory': '8',
            'sec-ch-dpr': '1',
            'sec-ch-ua': '".Not/A)Brand";v="99", "Google Chrome";v="103", "Chromium";v="103"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'document',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-site': 'none',
            'sec-fetch-user': '?1',
            'upgrade-insecure-requests': '1',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
            }

        payload = ""

        response = requests.request("GET", baseUrl+str(item), data=payload,  headers=headersList)
        item_doc = pq(response.text)
        return (item, item_doc("span#productTitle").text())

    def __parse_gzip_file__(self, path):
        g = gzip.open(path, 'rb')
        for line in g:
            yield json.loads(line)

    def __get_pandas_from_gzip__(self, path):
        i = 0
        df = {}
        for d in self.__parse_gzip_file__(path):
            df[i] = d
            i += 1
        return pd.DataFrame.from_dict(df, orient='index')

    def load_subset(self, subset_name) -> pd.DataFrame:
        """Carga los datos de una ciudad, quedandose con las columnas relevantes"""

        # Cargar reviews, renombrar columnas y ordenar por fecha para crear "reviewId"
        rev = self.__get_pandas_from_gzip__(f'{self.CONFIG["data_path"]}{subset_name}/data.json.gz')
        columns_name_dict = {"reviewerID": "userId", "overall": "rating", "asin": "itemId", "reviewText": "text", "summary": "title", "unixReviewTime": "date"}
        rev = rev.rename(columns=columns_name_dict).sort_values("date").reset_index(drop=True)
        rev.insert(0, "reviewId", range(len(rev)))

        # Cargar metadata
        meta = self.__get_pandas_from_gzip__(f'{self.CONFIG["data_path"]}{subset_name}/metadata.json.gz')
        columns_name_dict = {"asin": "itemId", "title": "name"}
        meta = meta.rename(columns=columns_name_dict).sort_values("date").reset_index(drop=True)
        meta = meta.astype({'name': 'str'})
        # meta = meta[~meta["name"].str.contains('getTime')]  # Eliminar algunos que tienen HTML en el título por error

        '''
        # Descargar los nombres erroneos
        meta["name_len"] = meta["name"].apply(len)
        meta["name_html"] = meta["name"].str.contains("<")
        dowload_items = meta[(meta.name_len<5) | (meta.name_html)]
        dowload_items = meta.itemId.values
        executor = ThreadPoolExecutor()
        dowload_results = list(tqdm(executor.map(self.__dowload_item_name__, dowload_items), total=len(dowload_items)))
        '''

        # Quedarse con columnas relevantes y casting de algunas
        rev = rev[['reviewId', 'userId', 'itemId', 'rating', 'date', 'text', 'title']]
        rev["rating"] = rev.rating * 10
        rev = rev.astype({'reviewId': 'int64', 'rating': 'int64'})

        # Concatenar items y reviews
        rev = rev.merge(meta[["itemId", "name"]], left_on="itemId", right_on="itemId", how="left")

        # Eliminar reviews vacías (que tengan NAN, pero sigue habiendo reviews con texto=="")
        rev = rev.loc[(~rev["text"].isna()) & (~rev["title"].isna())]

        return rev
