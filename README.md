# PBEL — Course Recommender

Content-based course recommendations with **TF-IDF + TruncatedSVD** (fit by you,
from scratch — not a pretrained model) and cosine KNN.

New courses are transformed through the frozen, already-fitted TF-IDF/SVD and
appended to the live store — **no retraining**.

## 1. Train on Kaggle (100k+ rows)

Use the notebook — do not download the raw dataset locally for training:

1. Open [`train_kaggle.ipynb`](train_kaggle.ipynb) on [Kaggle](https://www.kaggle.com/).
2. **Add Input → Datasets** and attach `hossaingh/udemy-courses`
   (`Course_info.csv`, ~210k distinct Udemy courses — real course listings,
   not review-duplicated rows).
3. Run all cells. The notebook:
   - asserts ≥ 100,000 rows
   - filters to English (if a `language` column exists and enough rows remain)
   - dedupes on title
   - builds course text from `title` + `headline` + `category`
   - fits TF-IDF → TruncatedSVD (the actual training step)
   - runs sanity-check searches so you can eyeball quality before exporting
4. Download from Output: `courses.csv`, `tfidf.pkl`, `svd.pkl`, `vectors.npy`, `report.json`.
5. Place all five files in this project root.

Our last run: 209,734 raw rows → 122,927 after English-filter + dedupe,
5000 TF-IDF features, 200 SVD components (31% explained variance). See
`report.json` for the exact numbers from that run.

## 2. Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional local re-fit (after you already have `courses.csv` from Kaggle, e.g.
you hand-edited it and want fresh vectors without re-running Kaggle):

```bash
python train.py
```

## 3. Run

```bash
# Terminal 1
python server.py          # API on :5000

# Terminal 2
streamlit run app.py      # UI on :8501
```

### API

| Method | Path | Body | Notes |
|--------|------|------|-------|
| POST | `/recommend` | `{text, k}` | Top-k similar courses (cosine, KNN fit fresh per query) |
| POST | `/add` | `{course_name, course_about, popularity_score, enrollment_count}` | Real-time add (transform + append, no retrain) |
| POST | `/delete` | `{id}` | Remove one entry (row + vector), everything else untouched |
| GET | `/entries` | — | List pool |
| GET | `/health` | — | Liveness |

## Project layout

```
train_kaggle.ipynb   # Kaggle training + sanity checks + export (primary path)
train.py             # Local re-fit from courses.csv (secondary/optional path)
data_store.py        # Vector store + real-time add/delete, TF-IDF/SVD transform-only
server.py            # Flask API
app.py               # Streamlit UI
requirements.txt
report.json          # Metrics from the last Kaggle training run
```

## Notes

- `tfidf.pkl` / `svd.pkl` are fit once and never refit at runtime — new entries
  are only ever `.transform()`'d through them. This is what makes add/delete
  instant and retrain-free.
- KNN itself is **not** persisted — `data_store.py` fits a fresh
  `NearestNeighbors` on the current vector pool at query time, so search always
  reflects the latest add/delete state without any separate index-rebuild step.
- Queries with no overlap with the trained vocabulary (out-of-vocabulary) return
  no results rather than misleading near-zero-similarity matches.
- **Known limitation:** TF-IDF vocabulary is frozen at training time. A newly
  added course is only findable by queries that share words with the *trained*
  corpus — if every word in a new entry is genuinely novel (not seen during
  training), it won't surface in search until the next full retrain. On the
  real ~123k-course Udemy vocabulary this is unlikely for normal course
  descriptions, but worth knowing if you add something in a very different
  domain than what was trained on.
- Concurrent writes to `store.csv` / `vectors.npy` are serialized with a
  `FileLock` in `data_store.py`.
- User-added courses get ids like `user-<uuid>`; seed rows from the Kaggle
  export use `seed-<course_id>`.