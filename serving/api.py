"""
FastAPI serving API.

Endpoints:
    GET  /health
    POST /recommend
"""
import logging
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from fastapi import FastAPI, HTTPException

from config_loader import load_config
from data.schema import RecommendRequest, RecommendResponse, ItemScore
from features.dense_features import DenseFeatureStore
from models.two_tower import build_model
from models.ranker import build_ranker
from retrieval.faiss_index import query_index, load_index

_cfg = None
_index = None
_item_embeddings = None
_user_embeddings = None
_id_maps = None
_ranker = None
_features = None        # DenseFeatureStore — shared with training/eval
_inv_item_map = None    # internal item index -> original dataset item_id
_cache: dict = {}   # user_id -> (RecommendResponse, expiry_float)

logger = logging.getLogger("recsys.api")

app = FastAPI(title="Short-Video RecSys", version="1.0.0")


def _load_artifacts(cfg: dict) -> None:
    """
    Load all ML artifacts into module-level globals.

    Called once at startup — all objects are kept in process memory for the
    lifetime of the server.  This avoids per-request disk I/O and keeps
    p99 latency low.

    Artifacts loaded:
        _id_maps:         user_id_map, item_id_map, n_users, n_items
        _item_embeddings: (n_items, embedding_dim) numpy array
        _user_embeddings: (n_users, embedding_dim) numpy array
        _index:           FAISS IndexFlatIP — used for candidate retrieval
        _ranker:          Trained MLPRanker in eval mode — used for re-ranking

    Sync function — called from the async startup event handler so that
    blocking file I/O does not run inside the async event loop.
    """
    global _index, _item_embeddings, _user_embeddings, _id_maps, _ranker
    global _features, _inv_item_map
    with open("datastore/processed/id_maps.pkl", "rb") as f:
        _id_maps = pickle.load(f)
    _item_embeddings = np.load("datastore/processed/item_embeddings.npy")
    _user_embeddings = np.load("datastore/processed/user_embeddings.npy")
    _index = load_index(cfg["retrieval"]["index_path"])

    # Inverted once at startup, not per request — rebuilding it inside the
    # request path is O(n_items) of pure overhead on every call.
    _inv_item_map = {v: k for k, v in _id_maps["item_id_map"].items()}

    # Same feature assembly as training and evaluation — see
    # features/dense_features.py.
    _features = DenseFeatureStore.load()
    ranker = build_ranker(cfg, _features.user_dense_dim, _features.item_dense_dim)
    ranker.load_state_dict(torch.load("datastore/processed/ranker_model.pt", map_location="cpu", weights_only=True))
    ranker.eval()
    _ranker = ranker


@app.on_event("startup")
async def startup():
    global _cfg
    _cfg = load_config()

    log_dir = Path(_cfg["monitoring"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "api.log"),
            logging.StreamHandler(),
        ],
    )

    _load_artifacts(_cfg)
    logger.info("API startup complete — %d users, %d items", _id_maps["n_users"], _id_maps["n_items"])


def _get_cache(user_id: int) -> RecommendResponse | None:
    """
    Return a cached RecommendResponse for `user_id` if it has not expired.

    Cache entries are (response, expiry_timestamp) tuples stored in the
    module-level `_cache` dict.  TTL is read from cfg[serving][cache_ttl_seconds].
    Returns None on miss or expiry — caller must generate a fresh response.
    """
    entry = _cache.get(user_id)
    if entry and time.time() < entry[1]:
        return entry[0]
    return None


def _set_cache(user_id: int, response: RecommendResponse) -> None:
    """
    Store a RecommendResponse in the in-process TTL cache.

    In production this would be replaced by a Redis call.  The in-process
    dict is sufficient for single-instance development/staging.
    """
    ttl = _cfg["serving"]["cache_ttl_seconds"]
    _cache[user_id] = (response, time.time() + ttl)


@app.get("/health")
def health() -> dict:
    """
    Liveness probe — returns {"status": "ok"} when the server is running.
    Used by load balancers and container orchestrators (e.g. Kubernetes) to
    determine whether the instance should receive traffic.
    """
    return {"status": "ok"}


@app.post("/recommend", responses={404: {"description": "Unknown user_id"}})
async def recommend(request: RecommendRequest) -> RecommendResponse:
    """
    Two-stage recommendation: retrieve top-K candidates → re-rank → return top-N.

    Stage 1 — Retrieval (ANN search):
        Look up the pre-computed user embedding, query the FAISS index for the
        top_k_recall (default 200) nearest item embeddings by cosine similarity.
        O(1) lookup + O(d * n_items) scan inside FAISS — typically < 5 ms.

    Stage 2 — Re-ranking (MLP ranker):
        For each candidate, build the feature vector
            [user_emb || item_emb || user_dense || item_dense]
        and run it through the ranker to predict watch_ratio ∈ [0, 1].
        Sort by predicted score descending, return the top request.top_k items.

    Caching:
        Results are cached per user_id with a configurable TTL.  On cache hit
        the full retrieval + ranking computation is skipped.

    Args:
        request: RecommendRequest with user_id and top_k.

    Returns:
        RecommendResponse with ranked item list, recall_size, and latency_ms.

    Raises:
        HTTPException(404): if user_id is not in the training ID map.
    """
    t0 = time.perf_counter()

    cached = _get_cache(request.user_id)
    if cached is not None:
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info("user=%d top_k=%d cache_hit=True latency_ms=%.1f",
                    request.user_id, request.top_k, latency_ms)
        return cached

    internal_uid = _id_maps["user_id_map"].get(request.user_id)
    if internal_uid is None:
        raise HTTPException(status_code=404, detail=f"Unknown user_id: {request.user_id}")

    user_emb = _user_embeddings[internal_uid]
    top_k_recall = _cfg["retrieval"]["top_k_recall"]
    _, item_indices = query_index(_index, user_emb, top_k=top_k_recall)

    # All candidates scored in a single batched forward pass — one call per
    # candidate is dominated by per-call PyTorch overhead at 200 candidates.
    x = torch.from_numpy(
        _features.build_batch(user_emb, _item_embeddings, internal_uid, item_indices)
    )
    with torch.no_grad():
        rank_scores = _ranker.predict(x).squeeze(-1).numpy()

    ranked = [(int(iid), float(s)) for iid, s in zip(item_indices, rank_scores)]
    ranked.sort(key=lambda t: t[1], reverse=True)
    top = ranked[:request.top_k]

    recommendations = [
        ItemScore(item_id=_inv_item_map.get(iid, iid), score=score, rank=rank + 1)
        for rank, (iid, score) in enumerate(top)
    ]

    latency_ms = (time.perf_counter() - t0) * 1000
    response = RecommendResponse(
        user_id=request.user_id,
        recommendations=recommendations,
        recall_size=len(item_indices),
        latency_ms=latency_ms,
    )
    _set_cache(request.user_id, response)
    logger.info("user=%d top_k=%d cache_hit=False latency_ms=%.1f",
                request.user_id, request.top_k, latency_ms)
    return response
