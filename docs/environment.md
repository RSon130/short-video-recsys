# Environment Setup

## Strategy
- **Local dev:** conda (GPU-enabled), env name `recsys`
- **Serving:** Docker with NVIDIA GPU passthrough

## Key Decisions
- Python 3.11 — 3.12 has known friction with PyTorch/FAISS/Pydantic v2
- `faiss-gpu` via conda-forge locally; pip wheel in Docker (pytorch base image handles it)
- `requirements.txt` is the single source of truth for package versions
- `environment.yml` installs pip packages from `requirements-dev.txt`
- Docker base image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`
- CUDA 12.1 (`pytorch-cuda=12.1` in conda)
- `device: auto` config key handles GPU detection at runtime — no code changes needed

## Files
| File | Purpose |
|---|---|
| `requirements.txt` | Production deps (faiss-gpu, torch, fastapi, pydantic, redis, etc.) |
| `requirements-dev.txt` | Extends requirements.txt + pytest, ruff, black, ipykernel |
| `environment.yml` | Conda env `recsys` with GPU FAISS + pip deps |
| `Dockerfile` | Lean GPU serving image |
| `docker-compose.yml` | Tier 1 serving with NVIDIA device reservation; Redis commented out for Tier 2 |
| `.dockerignore` | Excludes .git, __pycache__, datastore/raw/, .env, checkpoints/, logs/ |

## Bootstrap: Local Environment
```powershell
conda env create -f environment.yml
conda activate recsys
python -c "import torch, faiss, fastapi, pydantic; print('OK')"
```

## Bootstrap: Docker (requires serving/api.py)
```powershell
docker build -t recsys:local .
docker compose up
```

## Tier 2 Notes
- Uncomment `redis` service in `docker-compose.yml`
- For GCP Cloud Run, swap base image to match target CUDA version
