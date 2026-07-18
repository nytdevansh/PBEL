"""
data_store.py — persistent vector store for the Course Recommender.

Model: TF-IDF (fit by you) -> TruncatedSVD (fit by you). Both are FROZEN after
training in train_kaggle.ipynb (or train.py) — new courses are only ever
.transform()'d through them, never refit. This is what makes add_entry() an
O(1)-ish, no-retrain operation.

Performance notes (this file was optimized to fix search hangs on 100k+ rows):
  - The row table and vector matrix are kept in memory (self._df / self._vec)
    after the first load. Every search previously did `pd.read_csv` on a
    123k-row CSV *and* rebuilt a sklearn NearestNeighbors tree from scratch —
    on every request. That's now done once at startup and updated in-place
    on add/delete, so search no longer touches disk at all.
  - search() computes cosine similarity as a single normalized dot product
    (matrix @ vector) instead of constructing a fresh NearestNeighbors object
    per query. For our dense (n, 200) float32 matrix this is a single fast
    BLAS call rather than sklearn's per-call object-construction overhead.
  - Disk writes (CSV + npy) still happen on add/delete for durability, and are
    still serialized with a FileLock, but reads no longer round-trip through
    disk.
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from filelock import FileLock

_HERE = Path(__file__).parent
STORE_CSV = _HERE / "store.csv"
VECTORS_NPY = _HERE / "vectors.npy"
LOCK_FILE = _HERE / "store.lock"
COURSES_CSV = _HERE / "courses.csv"
TFIDF_PKL = _HERE / "tfidf.pkl"
SVD_PKL = _HERE / "svd.pkl"

STORE_COLS = [
    "id",
    "course_id",
    "course_name",
    "course_about",
    "popularity_score",
    "enrollment_count",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


class DataStore:
    """Single shared instance — instantiate once in server.py."""

    def __init__(self) -> None:
        for p in (TFIDF_PKL, SVD_PKL):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing {p.name}. Run train_kaggle.ipynb on Kaggle "
                    "(hossaingh/udemy-courses) and place tfidf.pkl, svd.pkl, "
                    "courses.csv, vectors.npy in the project root."
                )
        log.info("Loading fitted tfidf.pkl and svd.pkl (no refitting) ...")
        self.tfidf = joblib.load(TFIDF_PKL)
        self.svd = joblib.load(SVD_PKL)

        # In-memory cache. Guarded by a simple RLock for thread-safety
        # (Flask's dev server can be run with threaded=True).
        self._mem_lock = threading.RLock()
        self._df: pd.DataFrame
        self._vec: np.ndarray
        self._vec_norm: np.ndarray  # L2-normalized copy, cached for cosine sim

        self._bootstrap_if_needed()
        self._df, self._vec = self._read_from_disk()
        self._rebuild_norm_cache()
        log.info("DataStore ready - %d entries in pool (in-memory).", len(self._df))

    # ---------- embedding ----------

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Transform text through the ALREADY-FITTED tfidf + svd. Never refits."""
        tfidf_vecs = self.tfidf.transform(texts)
        return self.svd.transform(tfidf_vecs).astype(np.float32)

    def _is_oov(self, text: str) -> bool:
        """True if the text shares no vocabulary with the trained TF-IDF corpus."""
        return self.tfidf.transform([text]).nnz == 0

    # ---------- bootstrap / disk I/O (only on startup + writes) ----------

    def _bootstrap_if_needed(self) -> None:
        if STORE_CSV.exists() and VECTORS_NPY.exists():
            return

        if not COURSES_CSV.exists():
            raise FileNotFoundError(
                f"Missing {COURSES_CSV.name}. Run train_kaggle.ipynb on Kaggle "
                "and place courses.csv (+ vectors.npy) in the project root."
            )

        log.info("Bootstrap: building store from courses.csv ...")
        df = pd.read_csv(COURSES_CSV)
        required = {"course_id", "course_name", "course_about"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"courses.csv missing columns: {sorted(missing)}")
        for col in ("popularity_score", "enrollment_count"):
            if col not in df.columns:
                df[col] = 0

        # Prefer precomputed vectors from train_kaggle.ipynb (avoids re-transforming
        # 100k+ rows locally, which is slow and pointless since they're already fitted).
        if VECTORS_NPY.exists():
            vectors = np.load(VECTORS_NPY).astype(np.float32)
            if len(vectors) != len(df):
                log.warning(
                    "vectors.npy length mismatch (%d vs %d) - re-transforming "
                    "via fitted tfidf/svd.", len(vectors), len(df),
                )
                texts = (df["course_name"].fillna("") + " " + df["course_about"].fillna("")).tolist()
                vectors = self._embed(texts)
        else:
            log.warning("vectors.npy missing - transforming all courses locally.")
            texts = (df["course_name"].fillna("") + " " + df["course_about"].fillna("")).tolist()
            vectors = self._embed(texts)

        store = df[["course_id", "course_name", "course_about",
                     "popularity_score", "enrollment_count"]].copy()
        if "id" in df.columns:
            store.insert(0, "id", df["id"])
        else:
            store.insert(0, "id", "seed-" + df["course_id"].astype(str))

        with FileLock(LOCK_FILE):
            store[STORE_COLS].to_csv(STORE_CSV, index=False)
            np.save(VECTORS_NPY, vectors)

        log.info("Bootstrap complete - %d entries written.", len(store))

    def _read_from_disk(self) -> tuple[pd.DataFrame, np.ndarray]:
        return pd.read_csv(STORE_CSV), np.load(VECTORS_NPY).astype(np.float32)

    def _write_to_disk(self, df: pd.DataFrame, vec: np.ndarray) -> None:
        df[STORE_COLS].to_csv(STORE_CSV, index=False)
        np.save(VECTORS_NPY, vec)

    def _rebuild_norm_cache(self) -> None:
        """Precompute L2-normalized vectors so search is a single dot product."""
        if len(self._vec) == 0:
            self._vec_norm = self._vec
            return
        norms = np.linalg.norm(self._vec, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero for all-zero vectors
        self._vec_norm = (self._vec / norms).astype(np.float32)

    # ---------- public API (same signatures as before) ----------

    def search(self, text: str, k: int = 10) -> list[dict]:
        """Cosine search over the current in-memory pool, via normalized dot
        product + argpartition top-k (no per-request tree construction, no
        disk I/O). Returns [] on OOV query instead of misleading near-zero
        similarity matches."""
        if self._is_oov(text):
            return []

        query_vec = self._embed([text])[0]
        qnorm = np.linalg.norm(query_vec)
        if qnorm == 0:
            return []
        query_vec = query_vec / qnorm

        with self._mem_lock:
            df = self._df
            vec_norm = self._vec_norm

        n = len(df)
        if n == 0:
            return []

        k = min(k, n)
        sims = vec_norm @ query_vec  # cosine similarity, shape (n,)

        if k < n:
            top_idx = np.argpartition(-sims, k - 1)[:k]
            top_idx = top_idx[np.argsort(-sims[top_idx])]
        else:
            top_idx = np.argsort(-sims)

        results = []
        for idx in top_idx:
            row = df.iloc[idx]
            results.append({
                "id": row["id"],
                "course_name": row["course_name"],
                "course_about": row["course_about"],
                "popularity_score": float(row["popularity_score"]),
                "enrollment_count": int(row["enrollment_count"]),
                "similarity": round(float(sims[idx]), 4),
            })
        return results

    def add_entry(
        self,
        course_name: str,
        course_about: str,
        popularity_score: float = 0.0,
        enrollment_count: int = 0,
    ) -> str:
        """Transform a new course through the frozen tfidf+svd and append it to
        the live pool. No retraining — instant and searchable on the next query."""
        uid = f"user-{uuid.uuid4().hex[:12]}"
        content = f"{course_name} {course_about}"
        new_vec = self._embed([content])

        new_row = pd.DataFrame([{
            "id": uid,
            "course_id": uid,
            "course_name": course_name,
            "course_about": course_about,
            "popularity_score": popularity_score,
            "enrollment_count": enrollment_count,
        }])

        with self._mem_lock, FileLock(LOCK_FILE):
            df = pd.concat([self._df, new_row], ignore_index=True)
            vec = np.vstack([self._vec, new_vec]).astype(np.float32)
            self._write_to_disk(df, vec)
            self._df, self._vec = df, vec
            self._rebuild_norm_cache()

        log.info("Added entry %s.", uid)
        return uid

    def delete_entry(self, entry_id: str) -> bool:
        """Remove exactly one entry (row + its vector). Everything else untouched."""
        with self._mem_lock, FileLock(LOCK_FILE):
            df = self._df
            mask = df["id"] == entry_id
            if not mask.any():
                return False
            idx = int(df.index[mask][0])
            df = df.drop(index=idx).reset_index(drop=True)
            vec = np.delete(self._vec, idx, axis=0).astype(np.float32)
            self._write_to_disk(df, vec)
            self._df, self._vec = df, vec
            self._rebuild_norm_cache()

        log.info("Deleted entry %s.", entry_id)
        return True

    def get_entries(self) -> list[dict]:
        with self._mem_lock:
            df = self._df
            cols = ["id", "course_name", "popularity_score", "enrollment_count"]
            cols = [c for c in cols if c in df.columns]
            return df[cols].to_dict(orient="records")


if __name__ == "__main__":
    import time

    ds = DataStore()
    print("\n--- Search timing ---")
    t0 = time.time()
    for r in ds.search("python machine learning", k=3):
        print(f"  {r['similarity']:.3f}  {r['course_name'][:60]}")
    print(f"  search() took {time.time() - t0:.4f}s")

    print("\n--- Add / find / delete ---")
    uid = ds.add_entry(
        course_name="Python for Data Engineers",
        course_about="python programming for building data pipelines and ETL workflows",
        popularity_score=0.9,
        enrollment_count=500,
    )
    print(f"  Added: {uid}")
    found = any(r["id"] == uid for r in ds.search("python data pipelines", k=10))
    print(f"  Found in search: {found}")
    print(f"  Deleted: {ds.delete_entry(uid)}")

    print("\n--- OOV handling ---")
    results = ds.search("underwater basket weaving buoyancy knots", k=3)
    print(f"  OOV query results: {results} (expect [])")
