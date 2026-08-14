"""
Configuration loading: base.yaml + a dataset-specific override.

Previously this lived in features/engineer.py (an odd home for config loading)
and was duplicated inside serving/api.py. Both copies used dict.update(), a
*shallow* merge, so a top-level key present in the override replaced the entire
base block rather than being merged into it. Because config/kuairec.yaml defines
a `data:` section, that silently discarded every other key under `data:` —
train_ratio, val_ratio, raw_dir, processed_dir — and the pipeline died with
KeyError: 'train_ratio' on the first real run.

One implementation, deep merge, used by every entry point.
"""
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent
BASE_CONFIG = "config/base.yaml"


def deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge `override` into `base`, returning a new dict.

    Nested mappings are merged key by key; any non-mapping value in `override`
    replaces its counterpart. `base` is not mutated.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config/kuairec.yaml") -> dict:
    """
    Load base.yaml and merge a dataset-specific override on top.

    Args:
        config_path: Override config path, relative to the project root
                     (e.g. "config/kuairec.yaml").

    Returns:
        Merged config dict.
    """
    with open(PROJECT_ROOT / BASE_CONFIG) as f:
        base = yaml.safe_load(f)
    with open(PROJECT_ROOT / config_path) as f:
        override = yaml.safe_load(f)
    return deep_merge(base, override)
