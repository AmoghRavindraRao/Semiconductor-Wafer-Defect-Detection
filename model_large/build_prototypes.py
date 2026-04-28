"""
build_prototypes.py
Build per-class centroids and a cosine FAISS index (IndexFlatIP on
L2-normalized embeddings) from the labeled embedding banks.

PRD §9.1, §9.2. Input banks are expected to already be L2-normalized; we
re-normalize defensively.

Outputs:
  embeddings/centroids.npy           (9, 256) float32, L2-normalized
  embeddings/faiss_index.bin         IndexFlatIP over labeled_embeddings_norm
  embeddings/faiss_labels.npy        labels aligned to the index rows (int64)
  embeddings/faiss_sources.npy       0=small_train, 1=pkl_labeled

Note: We concatenate small-train and pkl-labeled for the FAISS index so kNN
can draw on both sources. Centroids are also built from this combined set.
Val/test are NOT indexed - they are held out for threshold tuning and eval.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils import NUM_CLASSES, PATHS, get_logger


def l2norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb_dir", type=Path, default=PATHS["emb"])
    args = parser.parse_args()

    logger = get_logger("protos", PATHS["results"] / "build_prototypes.log")

    train_emb = np.load(args.emb_dir / "train_embeddings.npy")
    train_y = np.load(args.emb_dir / "train_labels.npy")
    pkl_emb = np.load(args.emb_dir / "pkl_labeled_embeddings.npy")
    pkl_y = np.load(args.emb_dir / "pkl_labeled_labels.npy")

    logger.info("train: %s pkl_labeled: %s", train_emb.shape, pkl_emb.shape)

    labeled_emb = np.concatenate([train_emb, pkl_emb], axis=0).astype(np.float32)
    labeled_y = np.concatenate([train_y, pkl_y], axis=0).astype(np.int64)
    sources = np.concatenate([
        np.zeros(len(train_emb), dtype=np.int64),
        np.ones(len(pkl_emb), dtype=np.int64),
    ], axis=0)

    labeled_norm = l2norm(labeled_emb)

    # --- Centroids (§9.1) -------------------------------------------------
    centroids = np.zeros((NUM_CLASSES, labeled_norm.shape[1]), dtype=np.float32)
    for c in range(NUM_CLASSES):
        mask = labeled_y == c
        count = int(mask.sum())
        if count == 0:
            logger.warning("class %d has 0 samples; centroid will be zero-vector", c)
            continue
        mu = labeled_norm[mask].mean(axis=0)
        n = np.linalg.norm(mu)
        centroids[c] = mu / max(n, 1e-12)
        logger.info("centroid %d: %d samples", c, count)

    np.save(args.emb_dir / "centroids.npy", centroids)
    logger.info("wrote %s (shape=%s)", args.emb_dir / "centroids.npy", centroids.shape)

    # --- FAISS cosine index (§9.2) ---------------------------------------
    dim = labeled_norm.shape[1]
    try:
        import faiss
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine
        index.add(labeled_norm)
        faiss.write_index(index, str(args.emb_dir / "faiss_index.bin"))
        logger.info("wrote %s (ntotal=%d dim=%d)",
                    args.emb_dir / "faiss_index.bin", index.ntotal, dim)
    except ImportError:
        logger.warning(
            "faiss not available - saving raw labeled_norm for brute-force kNN"
        )
        np.save(args.emb_dir / "faiss_labeled_norm.npy", labeled_norm)

    np.save(args.emb_dir / "faiss_labels.npy", labeled_y)
    np.save(args.emb_dir / "faiss_sources.npy", sources)
    logger.info("wrote %s  (N=%d)", args.emb_dir / "faiss_labels.npy", len(labeled_y))


if __name__ == "__main__":
    main()
