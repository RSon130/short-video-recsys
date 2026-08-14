from config_loader import deep_merge, load_config


def test_deep_merge_preserves_sibling_keys():
    """
    The bug this guards against: dict.update() let an override's top-level key
    replace the entire base block, silently dropping its siblings.
    """
    base = {"data": {"train_ratio": 0.8, "val_ratio": 0.1, "source": "base"}}
    override = {"data": {"source": "kuairec"}}

    merged = deep_merge(base, override)

    assert merged["data"]["source"] == "kuairec"
    assert merged["data"]["train_ratio"] == 0.8
    assert merged["data"]["val_ratio"] == 0.1


def test_deep_merge_recurses_into_nested_mappings():
    base = {"training": {"ranking": {"epochs": 30, "lr": 5e-4}}}
    override = {"training": {"ranking": {"epochs": 1}}}

    merged = deep_merge(base, override)

    assert merged["training"]["ranking"] == {"epochs": 1, "lr": 5e-4}


def test_deep_merge_does_not_mutate_base():
    base = {"a": {"b": 1}}
    deep_merge(base, {"a": {"b": 2}})
    assert base == {"a": {"b": 1}}


def test_deep_merge_replaces_non_mapping_values():
    base = {"k_values": [5, 10, 20]}
    merged = deep_merge(base, {"k_values": [10]})
    assert merged["k_values"] == [10]


def test_real_config_keeps_split_ratios():
    """config/kuairec.yaml defines `data:` — the keys it omits must survive."""
    cfg = load_config("config/kuairec.yaml")

    assert cfg["data"]["source"] == "kuairec"
    assert cfg["data"]["train_ratio"] == 0.8
    assert cfg["data"]["val_ratio"] == 0.1
    assert cfg["data"]["processed_dir"] == "datastore/processed"
    assert "kuairec" in cfg["data"]


def test_real_config_exposes_keys_every_entry_point_reads():
    cfg = load_config("config/kuairec.yaml")

    assert cfg["ranking"]["hidden_dims"]
    assert cfg["serving"]["cache_ttl_seconds"]
    assert cfg["retrieval"]["top_k_recall"]
    assert cfg["evaluation"]["k_values"]


def test_numeric_hyperparameters_load_as_numbers():
    """
    YAML 1.1 only reads an exponent as a float when written 1.0e-3; bare 1e-3
    loads as a str and blows up inside the optimiser with an unhelpful
    TypeError. Guard every numeric hyperparameter that reaches PyTorch.
    """
    cfg = load_config("config/kuairec.yaml")

    for stage in ("retrieval", "ranking"):
        params = cfg["training"][stage]
        assert isinstance(params["lr"], float), f"{stage}.lr is {type(params['lr'])}"
        assert isinstance(params["weight_decay"], float)
        assert isinstance(params["epochs"], int)
        assert isinstance(params["batch_size"], int)

    assert isinstance(cfg["two_tower"]["dropout"], float)
    assert isinstance(cfg["ranking"]["dropout"], float)
