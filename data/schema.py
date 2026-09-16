"""
Canonical data schemas (Pydantic v2).

All data sources must produce DataFrames whose columns match these names.
The mapping from source-specific column names is handled in each loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# DataFrame column name constants — import these everywhere instead of strings
# ---------------------------------------------------------------------------

class Cols:
    """Canonical column names shared across the entire project."""
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    TIMESTAMP = "timestamp"
    WATCH_RATIO = "watch_ratio"   # float in [0, 1]
    LIKE = "like"                 # int 0/1
    COMMENT = "comment"           # int 0/1
    SHARE = "share"               # int 0/1
    FOLLOW = "follow"             # int 0/1

    # Derived interaction signals
    ENGAGEMENT = "engagement"     # composite score
    LABEL_WATCH = "label_watch"   # regression target
    LABEL_LIKE = "label_like"     # binary classification target

    # Feature columns
    USER_EMB = "user_emb"         # np.ndarray stored as object column
    ITEM_EMB = "item_emb"

    # Session columns
    SESSION_ITEMS = "session_items"
    SESSION_WATCHES = "session_watches"


# ---------------------------------------------------------------------------
# Pydantic models — used for API request/response validation
# ---------------------------------------------------------------------------

class RecommendRequest(BaseModel):
    user_id: int
    session_items: List[int] = Field(default_factory=list, max_length=50)
    top_k: int = Field(default=20, ge=1, le=100)
    strategy: str = Field(default="full", pattern="^(full|recall_only|rank_only)$")


class ItemScore(BaseModel):
    item_id: int
    score: float
    rank: int


class RecommendResponse(BaseModel):
    user_id: int
    recommendations: List[ItemScore]
    recall_size: int
    latency_ms: float


class KuaiRandRequest(BaseModel):
    user_id: int
    top_k: int = Field(default=20, ge=1, le=100)
    exclude_seen: bool = True


class KuaiRandResponse(BaseModel):
    user_id: int
    recommendations: List[ItemScore]
    scorer: str
    scorer_note: str
    catalogue_size: int
    excluded_seen: int
    known_user: bool
    latency_ms: float


class InteractionEvent(BaseModel):
    user_id: int
    item_id: int
    watch_ratio: float = Field(ge=0.0, le=1.0)
    like: int = Field(default=0, ge=0, le=1)
    comment: int = Field(default=0, ge=0, le=1)
    share: int = Field(default=0, ge=0, le=1)
    follow: int = Field(default=0, ge=0, le=1)
    timestamp: Optional[int] = None

    @field_validator("watch_ratio")
    @classmethod
    def clip_watch_ratio(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Lightweight dataclasses used internally (no validation overhead)
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


@dataclass
class FeatureStore:
    """Holds processed feature tables; passed between pipeline stages."""
    interactions: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame
    user_id_map: dict = field(default_factory=dict)   # raw_id -> int index
    item_id_map: dict = field(default_factory=dict)
    n_users: int = 0
    n_items: int = 0
