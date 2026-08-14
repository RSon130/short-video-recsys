import numpy as np
import pandas as pd
import pytest

from features.engineer import normalize_dense_features


def test_extreme_scales_are_brought_into_range():
    """
    The failure this prevents: raw KuaiRec item features reach 2.6e11 while the
    tower embeddings are L2-normalised to ~0.1. Feeding both to the ranker
    saturated its sigmoid and it predicted a constant for every input.
    """
    df = pd.DataFrame({
        "item_id": [0, 1, 2, 3],
        "plays": [1.0, 1e6, 1e9, 2.6e11],
        "small": [0.1, 0.2, 0.3, 0.4],
    })

    out = normalize_dense_features(df, "item_id")
    values = out.drop(columns=["item_id"]).to_numpy()

    assert np.abs(values).max() < 5.0


def test_columns_are_centred_and_scaled():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "user_id": range(200),
        "a": rng.lognormal(5, 2, 200),
        "b": rng.integers(0, 2000, 200).astype(float),
    })

    out = normalize_dense_features(df, "user_id").drop(columns=["user_id"])

    assert out.mean().abs().max() < 1e-5
    assert out.std().sub(1.0).abs().max() < 0.05


def test_constant_column_becomes_zero_not_nan():
    df = pd.DataFrame({"item_id": [0, 1, 2], "flat": [7.0, 7.0, 7.0]})

    out = normalize_dense_features(df, "item_id")

    assert out["flat"].tolist() == [0.0, 0.0, 0.0]
    assert not out.isna().any().any()


def test_id_column_is_preserved_unchanged():
    df = pd.DataFrame({"user_id": [10, 20, 30], "x": [1.0, 2.0, 3.0]})

    out = normalize_dense_features(df, "user_id")

    assert out["user_id"].tolist() == [10, 20, 30]
    assert list(out.columns) == ["user_id", "x"]


def test_monotonicity_is_preserved():
    """Normalisation may rescale, but it must not reorder."""
    df = pd.DataFrame({"item_id": [0, 1, 2, 3], "x": [1.0, 10.0, 100.0, 1000.0]})

    out = normalize_dense_features(df, "item_id")["x"].to_numpy()

    assert np.all(np.diff(out) > 0)


def test_handles_zeros_and_negatives():
    df = pd.DataFrame({"item_id": [0, 1, 2], "x": [-100.0, 0.0, 100.0]})

    out = normalize_dense_features(df, "item_id")

    assert not out.isna().any().any()
    assert out["x"].iloc[0] < out["x"].iloc[1] < out["x"].iloc[2]


def test_output_is_float32():
    df = pd.DataFrame({"item_id": [0, 1], "x": [1.0, 2.0]})

    out = normalize_dense_features(df, "item_id")

    assert out["x"].dtype == np.float32
