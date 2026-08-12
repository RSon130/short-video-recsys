# Short-Video RecSys

A production-grade recommendation system for short-video feeds, modelled on the YouTube Shorts / TikTok architecture. Built in three tiers — each independently shippable and resume-worthy.

## Architecture

```
Request
  └─ Stage 1 · Retrieval   Two-Tower (BPR) + FAISS IndexFlatIP → top-200 candidates
  └─ Stage 2 · Ranking     MLP Ranker (MSE on watch_ratio)     → top-20 scored items
  └─ Serving               FastAPI + in-memory TTL cache
```

| Component | Implementation | Notes |
|-----------|---------------|-------|
| Retrieval | Two-tower neural net (BPR loss) | L2-normalised embeddings, cosine ANN |
| Index | FAISS `IndexFlatIP` | Exact IP search, <5 ms on 1K items |
| Ranker | MLP (3 layers, sigmoid output) | Predicts watch_ratio ∈ [0, 1] |
| Serving | FastAPI + uvicorn | TTL in-memory cache |
| Dataset | KuaiRec `small_matrix` (1K×1K) | ~350K interactions after cold-start filter |

## Quickstart

### Local (conda)

```bash
conda env create -f environment.yml
conda activate recsys

# 1. Download data
python scripts/download_kuairec.py --subset small

# 2. Feature engineering
python features/engineer.py

# 3. Train retrieval model (~5 min on CPU)
python training/train_retrieval.py

# 4. Build FAISS index
python scripts/build_index.py

# 5. Train ranker
python training/train_ranking.py

# 6. Evaluate
python scripts/evaluate.py

# 7. Start API
uvicorn serving.api:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker-compose up --build
```

Then send a request:

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 20}'
```

## Evaluation Results (Tier 1 — small_matrix)

| Metric | Value |
|--------|-------|
| Recall@5 | — |
| Recall@10 | — |
| Recall@20 | — |
| NDCG@5 | — |
| NDCG@10 | — |
| NDCG@20 | — |
| Watch-time AUC (retrieval) | — |
| Watch-time AUC (ranker) | — |

> Numbers filled in after training completes.

## API

**`POST /recommend`**

```json
{
  "user_id": 42,
  "top_k": 20
}
```

Returns:

```json
{
  "user_id": 42,
  "recommendations": [{"item_id": 101, "score": 0.87, "rank": 1}, ...],
  "recall_size": 200,
  "latency_ms": 18.4
}
```

**`GET /health`** → `{"status": "ok"}`

## Project Structure

```
config/          # base.yaml + kuairec.yaml
data/            # loaders, schema, factory
features/        # engineer.py — temporal split, parquet export
models/          # two_tower.py, ranker.py
training/        # train_retrieval.py, train_ranking.py
retrieval/       # faiss_index.py
serving/         # FastAPI api.py
evaluation/      # metrics.py, ab_test.py
scripts/         # download_kuairec.py, build_index.py, evaluate.py
docs/            # system_design.md, learning_guide.md, progress.md
```

## Tier Roadmap

| Tier | What's added | Status |
|------|-------------|--------|
| 1 · Core ML | Two-tower + FAISS + MLP ranker + Docker | ✅ |
| 2 · Production | IVFFlat index, Redis cache, multi-task ranking, Cloud Run | — |
| 3 · ML Depth | InfoNCE loss, Transformer ranker, MMR diversity, Prefect | — |

## Design Decisions

- **BPR loss** (Tier 1) → InfoNCE with in-batch negatives (Tier 3): BPR is simple and robust; InfoNCE scales better with batch size.
- **IndexFlatIP** (Tier 1) → IVFFlat (Tier 2): exact search is fine at 1K items; IVFFlat reduces latency by ~10× at 100K+ items.
- **In-memory dict cache** (Tier 1) → Redis (Tier 2): single-instance dev cache; Redis enables multi-instance horizontal scaling.
- **Temporal split**: sorted by timestamp globally — prevents any future leakage into training.
