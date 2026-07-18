"""
app.py — Streamlit UI for the Course Recommender.

    python server.py          # terminal 1
    streamlit run app.py      # terminal 2
"""

import requests
import streamlit as st

API = "http://localhost:5000"

st.set_page_config(page_title="Course Recommender", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(160deg, #0b1220 0%, #152238 50%, #0f1a2e 100%); }
h1 { color: #e8eef8; font-weight: 700; letter-spacing: -0.02em; }
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
}
.card h4 { color: #e8eef8; margin: 0 0 0.4rem; font-size: 1rem; }
.card p { color: #94a3b8; margin: 0; font-size: 0.88rem; line-height: 1.55; }
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.28);
    color: #7dd3fc;
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.4rem;
}
.err {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    color: #fca5a5;
    padding: 0.8rem 1rem;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API}{endpoint}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="err">Cannot reach API — start <code>python server.py</code> on port 5000.</div>',
                    unsafe_allow_html=True)
        return None
    except Exception as exc:
        st.markdown(f'<div class="err">API error: {exc}</div>', unsafe_allow_html=True)
        return None


def api_get(endpoint: str):
    try:
        r = requests.get(f"{API}{endpoint}", timeout=15)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.markdown('<div class="err">Cannot reach API — start <code>python server.py</code> on port 5000.</div>',
                    unsafe_allow_html=True)
        return None
    except Exception as exc:
        st.markdown(f'<div class="err">API error: {exc}</div>', unsafe_allow_html=True)
        return None


st.title("Course Recommender")
st.caption("TF-IDF · TruncatedSVD · cosine KNN · trained from scratch · real-time add")

entries_data = api_get("/entries")
n_total = len(entries_data["entries"]) if entries_data else 0
n_user = sum(1 for e in (entries_data["entries"] if entries_data else []) if str(e["id"]).startswith("user-"))
st.write(f"**{n_total}** courses in pool · **{n_user}** user-added")

tab_search, tab_add, tab_manage = st.tabs(["Search", "Add course", "Manage"])

with tab_search:
    # Wrapped in a form: previously `if st.button(...) or query:` fired a full
    # /recommend API call on EVERY keystroke (Streamlit reruns the script on
    # every widget change), which is what caused the "hanging" search over a
    # 100k+ row pool. A form only submits on button click / Enter.
    with st.form("search_form"):
        col_q, col_k = st.columns([3, 1])
        with col_q:
            query = st.text_input("Query", placeholder="e.g. python machine learning", label_visibility="collapsed")
        with col_k:
            k = st.slider("Results", 3, 20, 10)
        submitted = st.form_submit_button("Recommend", type="primary")

    if submitted:
        if not query.strip():
            st.info("Type a search query above.")
        else:
            with st.spinner("Searching…"):
                data = api_post("/recommend", {"text": query, "k": k})
            if data is not None:
                results = data.get("results", [])
                if not results:
                    st.warning("No matches — try wording closer to how courses are usually titled/described.")
                else:
                    for i, r in enumerate(results, 1):
                        about = r["course_about"]
                        preview = about[:280] + ("…" if len(about) > 280 else "")
                        st.markdown(f"""
<div class="card">
  <h4>#{i} · {r['course_name']}</h4>
  <span class="badge">{r['similarity']*100:.1f}% match</span>
  <p>{preview}</p>
</div>
""", unsafe_allow_html=True)

with tab_add:
    st.write("New courses are transformed with the already-fitted TF-IDF + SVD "
              "(never refit) and become searchable immediately.")
    with st.form("add_form", clear_on_submit=True):
        name = st.text_input("Course name *")
        about = st.text_area("Description *", height=140)
        c1, c2 = st.columns(2)
        with c1:
            pop = st.number_input("Popularity (0–1)", 0.0, 1.0, 0.0, 0.01)
        with c2:
            enroll = st.number_input("Enrollment count", 0, value=0, step=1)
        submitted = st.form_submit_button("Add course", type="primary")

    if submitted:
        if not name.strip() or not about.strip():
            st.error("Name and description are required.")
        else:
            data = api_post("/add", {
                "course_name": name.strip(),
                "course_about": about.strip(),
                "popularity_score": pop,
                "enrollment_count": int(enroll),
            })
            if data and "id" in data:
                st.success(f"Added — id `{data['id']}`")
                st.rerun()

with tab_manage:
    if st.button("Refresh"):
        st.rerun()
    entries_data = api_get("/entries")
    if entries_data is None:
        st.stop()
    entries = entries_data.get("entries", [])
    source = st.radio("Filter", ["All", "Seed", "User-added"], horizontal=True)
    if source == "Seed":
        shown = [e for e in entries if str(e["id"]).startswith("seed-")]
    elif source == "User-added":
        shown = [e for e in entries if str(e["id"]).startswith("user-")]
    else:
        shown = entries

    st.write(f"Showing {len(shown)} of {len(entries)}")
    for entry in shown:
        eid = entry["id"]
        c1, c2 = st.columns([5, 1])
        with c1:
            tag = "seed" if str(eid).startswith("seed-") else "user"
            st.markdown(f"`{tag}` **{entry['course_name']}** · `{eid}`")
        with c2:
            if st.button("Delete", key=f"del_{eid}"):
                result = api_post("/delete", {"id": eid})
                if result and result.get("ok"):
                    st.rerun()
                elif result:
                    st.error(result.get("error", "Failed"))
