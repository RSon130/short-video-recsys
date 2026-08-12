"""
Data loader factory.

Usage:
    from data import get_loader
    loader = get_loader(cfg)
"""

from data.base import BaseDataLoader


def get_loader(cfg: dict) -> BaseDataLoader:
    source = cfg["data"]["source"]
    if source == "kuairec":
        from data.kuairec import KuaiRecLoader
        return KuaiRecLoader(cfg)
    raise ValueError(f"Unknown data source: '{source}'. Expected one of: ['kuairec']")
