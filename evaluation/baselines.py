"""
Reference systems to compare the trained pipeline against.

A metric reported on its own is not evidence. "NDCG@10 = 0.31" is unreadable
without knowing what a trivial system scores on the same split — and on a dense,
popularity-skewed dataset a trivial system can score surprisingly well. Every
headline number this project reports is stated against these baselines.

Two baselines, in increasing order of strength:

  popularity      Recommend the globally most-engaging items to everyone. No
                  personalisation whatsoever. If the trained model cannot beat
                  this, it has learned nothing useful.

  retrieval-only  Two-tower + FAISS, ranked by cosine similarity, with no
                  ranker stage. This isolates what the second stage adds, which
                  is the argument for having a two-stage architecture at all.

The retrieval-only baseline is produced inside scripts/evaluate.py, where the
index and embeddings are already loaded; only the popularity baseline needs
building here.
"""
import numpy as np
import pandas as pd

from data.schema import Cols


def popularity_ranking(train_df: pd.DataFrame, positive_threshold: float) -> np.ndarray:
    """
    Rank items by how often they were genuinely engaged with in training.

    Counting *positive* interactions rather than raw appearances matters on a
    fully-observed dataset: every item was shown to nearly every user, so raw
    interaction counts are almost uniform and measure exposure, not appeal.

    Args:
        train_df:           Training interactions.
        positive_threshold: Minimum watch_ratio to count as engagement.

    Returns:
        Item IDs ordered most- to least-popular.
    """
    positives = train_df.loc[train_df[Cols.WATCH_RATIO] >= positive_threshold]
    counts = positives[Cols.ITEM_ID].value_counts()
    return counts.index.to_numpy()


def popularity_recommendations(
    ranking: np.ndarray,
    users: list,
    top_k: int,
    seen: dict = None,
) -> list:
    """
    Produce a top-K popularity list per user.

    Args:
        ranking: Item IDs from popularity_ranking(), best first.
        users:   Users to generate recommendations for, in order.
        top_k:   Number of items per user.
        seen:    Optional {user_id: set(item_ids)} to exclude from the results
                 (items already consumed in training).

    Returns:
        List of per-user recommendation lists, aligned with `users`.
    """
    if seen is None:
        return [list(ranking[:top_k]) for _ in users]

    recommendations = []
    for uid in users:
        exclude = seen.get(uid, set())
        recommendations.append(
            [int(i) for i in ranking if int(i) not in exclude][:top_k]
        )
    return recommendations
