"""
server.py — Flask API for the Course Recommender.

  POST /recommend  {text, k}  -> {results: [...]}
  POST /add        {course_name, course_about, popularity_score, enrollment_count} -> {id}
  POST /delete     {id} -> {ok: true}
  GET  /entries    -> {entries: [...]}
  GET  /health     -> {status: "ok"}

Runs on port 5000.
"""

from flask import Flask, jsonify, request
from data_store import DataStore

app = Flask(__name__)
store = DataStore()


def _bad(msg: str, code: int = 400):
    return jsonify({"error": msg}), code


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/recommend")
def recommend():
    body = request.get_json(force=True, silent=True) or {}
    text = (body.get("text") or "").strip()
    k = int(body.get("k", 10))

    if not text:
        return _bad("text is required")
    if not (1 <= k <= 50):
        return _bad("k must be between 1 and 50")

    return jsonify({"results": store.search(text, k=k)})


@app.post("/add")
def add():
    body = request.get_json(force=True, silent=True) or {}
    course_name = (body.get("course_name") or "").strip()
    course_about = (body.get("course_about") or "").strip()

    if not course_name:
        return _bad("course_name is required")
    if not course_about:
        return _bad("course_about is required")

    try:
        popularity_score = float(body.get("popularity_score", 0.0))
        enrollment_count = int(body.get("enrollment_count", 0))
    except (TypeError, ValueError) as exc:
        return _bad(f"Invalid numeric field: {exc}")

    uid = store.add_entry(
        course_name=course_name,
        course_about=course_about,
        popularity_score=popularity_score,
        enrollment_count=enrollment_count,
    )
    return jsonify({"id": uid}), 201


@app.post("/delete")
def delete():
    body = request.get_json(force=True, silent=True) or {}
    entry_id = (body.get("id") or "").strip()
    if not entry_id:
        return _bad("id is required")
    if not store.delete_entry(entry_id):
        return _bad(f"Entry '{entry_id}' not found", 404)
    return jsonify({"ok": True})


@app.get("/entries")
def entries():
    return jsonify({"entries": store.get_entries()})


if __name__ == "__main__":
    # threaded=True lets health checks / other requests go through while a
    # search or add is in flight, instead of queuing behind it.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
