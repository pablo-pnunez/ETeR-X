# -*- coding: utf-8 -*-
"""Machine-independent configuration (paths).

Raw datasets are looked up under the directory given by the ``ETERX_DATA_DIR`` environment
variable (default: ``./datasets``), with the layout ``{root}/{amazon,tripadvisor/restaurants,
tripadvisor/pois}/...``.

Dataset and model checkpoint folders are named after an MD5 of their configuration, and that
configuration contains ``data_path``. To keep the hashes (and therefore the checkpoint folders and
``models/best_models.csv``) valid on any machine, hashes are computed with ``data_path``
rewritten relative to ``LEGACY_DATA_ROOT``, the location used when the checkpoints were created.
"""
import os

LEGACY_DATA_ROOT = "/media/nas/datasets"


def data_root():
    return os.environ.get("ETERX_DATA_DIR", "datasets").rstrip("/")


def data_path(*parts):
    """Path (with trailing slash) of a raw dataset, e.g. data_path("tripadvisor", "pois")."""
    return "/".join([data_root(), *parts]) + "/"


def canonical_for_md5(cfg):
    """Copy of a dataset config whose ``data_path`` is expressed under LEGACY_DATA_ROOT."""
    cfg = dict(cfg)
    p = cfg.get("data_path")
    root = data_root()
    if isinstance(p, str) and p.startswith(root + "/"):
        cfg["data_path"] = LEGACY_DATA_ROOT + p[len(root):]
    return cfg
