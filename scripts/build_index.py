"""
Build FAISS index from item embeddings.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --config config/kuairec.yaml
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from features.engineer import load_config
from retrieval import faiss_index


def main() -> None:
    """
    CLI entry point: load item embeddings → build FAISS index → save to disk.

    This script is the bridge between the retrieval training step (which exports
    item_embeddings.npy) and the serving API (which loads the index at startup).

    Pipeline position:
        train_retrieval.py  →  [item_embeddings.npy]  →  build_index.py
        →  [faiss.index]  →  serving/api.py

    The index only needs to be rebuilt when item embeddings change (i.e. after
    retraining the two-tower model).  It does NOT need to be rebuilt when the
    ranker is retrained.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/kuairec.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    project_root = Path(__file__).parent.parent
    embeddings = np.load(project_root / "data" / "processed" / "item_embeddings.npy")

    index = faiss_index.build_index(embeddings)

    path = project_root / cfg["retrieval"]["index_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss_index.save_index(index, path)

    n_items, dim = embeddings.shape
    print(f"Index built: {n_items} items, dim={dim}, saved to {path}")


if __name__ == "__main__":
    main()
