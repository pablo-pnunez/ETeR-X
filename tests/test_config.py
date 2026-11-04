import hashlib

from src import config


def test_data_path_uses_env(monkeypatch):
    monkeypatch.setenv("ETERX_DATA_DIR", "/somewhere/else/")
    assert config.data_path("tripadvisor", "pois") == "/somewhere/else/tripadvisor/pois/"


def test_default_data_root(monkeypatch):
    monkeypatch.delenv("ETERX_DATA_DIR", raising=False)
    assert config.data_path("amazon") == "datasets/amazon/"


def test_md5_is_independent_of_data_location(monkeypatch):
    """The hash must equal the one computed with the original (legacy) location."""
    legacy = {"dataset": "restaurants", "data_path": config.LEGACY_DATA_ROOT + "/tripadvisor/restaurants/"}
    expected = hashlib.md5(str(legacy).encode()).hexdigest()

    monkeypatch.setenv("ETERX_DATA_DIR", "/anywhere")
    moved = {"dataset": "restaurants", "data_path": config.data_path("tripadvisor", "restaurants")}
    assert hashlib.md5(str(config.canonical_for_md5(moved)).encode()).hexdigest() == expected


def test_canonical_does_not_mutate_input(monkeypatch):
    monkeypatch.setenv("ETERX_DATA_DIR", "/anywhere")
    cfg = {"data_path": "/anywhere/amazon/"}
    config.canonical_for_md5(cfg)
    assert cfg["data_path"] == "/anywhere/amazon/"
