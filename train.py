"""
train.py — local retrain of TF-IDF + TruncatedSVD from courses.csv.

Only needed if you want to rebuild the vectors locally after already having
courses.csv (e.g. you hand-edited it). The primary training path is
train_kaggle.ipynb, run on Kaggle against hossaingh/udemy-courses
(Course_info.csv, ~123k rows after cleaning — see report.json).

Usage:
    python train.py

Deletes store.csv so DataStore re-bootstraps on next server start.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

HERE = Path(__file__).parent
MAX_TFIDF_FEATURES = 5000
N_SVD_COMPONENTS = 200
RANDOM_STATE = 42


def main() -> None:
    csv_path = HERE / "courses.csv"
    if not csv_path.exists():
        raise SystemExit(
            "courses.csv not found.\n"
            "1) Run train_kaggle.ipynb on Kaggle with hossaingh/udemy-courses attached.\n"
            "2) Download courses.csv (+ tfidf.pkl, svd.pkl, vectors.npy) into this project root.\n"
            "This script only re-fits locally on whatever courses.csv is already here."
        )

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    for col in ("popularity_score", "enrollment_count"):
        if col not in df.columns:
            df[col] = 0
    print(f"  {len(df)} courses")

    content = (df["course_name"].fillna("") + " " + df["course_about"].fillna("")).str.strip()

    print("Fitting TF-IDF ...")
    tfidf = TfidfVectorizer(stop_words="english", max_features=MAX_TFIDF_FEATURES, min_df=2, max_df=0.8)
    tfidf_matrix = tfidf.fit_transform(content)
    print(f"  TF-IDF matrix: {tfidf_matrix.shape}")

    print("Fitting TruncatedSVD ...")
    svd = TruncatedSVD(n_components=N_SVD_COMPONENTS, random_state=RANDOM_STATE)
    vectors = svd.fit_transform(tfidf_matrix).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    print(f"  Reduced vectors: {vectors.shape}  explained_variance={explained:.3f}")

    print("Saving artifacts ...")
    joblib.dump(tfidf, HERE / "tfidf.pkl")
    joblib.dump(svd, HERE / "svd.pkl")
    np.save(HERE / "vectors.npy", vectors)

    report = {
        "source_file": str(csv_path),
        "rows": int(len(df)),
        "tfidf_features": int(tfidf_matrix.shape[1]),
        "svd_components": N_SVD_COMPONENTS,
        "svd_explained_variance": explained,
    }
    with open(HERE / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Report:", json.dumps(report, indent=2))

    for fname in ("store.csv", "store.lock"):
        p = HERE / fname
        if p.exists():
            p.unlink()
            print(f"  Deleted {fname}")

    print("\nDone. Start the server to bootstrap the live store.")


if __name__ == "__main__":
    main()