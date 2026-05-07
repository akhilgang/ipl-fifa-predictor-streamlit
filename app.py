"""
app.py  — IPL 2026 Match Predictor · Streamlit
Run:  streamlit run app.py
"""

import streamlit as st
import joblib
import json
import random
import os
from pathlib import Path
from collections import defaultdict
from copy import deepcopy
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IPL 2026 · AI Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
with open(Path(__file__).parent / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Artifact loader (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = Path(__file__).parent / "models"
    model    = joblib.load(base / "ipl_model.pkl")
    le_t1    = joblib.load(base / "ipl_le_t1.pkl")
    le_t2    = joblib.load(base / "ipl_le_t2.pkl")
    le_venue = joblib.load(base / "ipl_le_venue.pkl")
    return model, le_t1, le_t2, le_venue

@st.cache_data
def load_json(filename):
    p = Path(__file__).parent / "data" / filename
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# ── Feature builder ───────────────────────────────────────────────────────────
def _enc(le, val, fallback=0):
    try:    return int(le.transform([val])[0])
    except: return fallback

IPL_STAGE_W = {"league": 1, "qualifier": 2, "eliminator": 2, "final": 3}

def build_features(t1, t2, venue, stage, t1_wr, t2_wr, model, le_t1, le_t2, le_venue):
    sw = IPL_STAGE_W.get(stage.lower(), 1)
    return [[
        _enc(le_t1, t1), _enc(le_t2, t2),
        1, 1,
        _enc(le_venue, venue) if venue else 0,
        sw,
        t1_wr, t2_wr,
        0, 0,
        t1_wr - t2_wr, 0,
        0.5, 2026,
    ]]

def predict_match(t1, t2, venue, stage, t1_wr, t2_wr):
    model, le_t1, le_t2, le_venue = load_artifacts()
    feats   = build_features(t1, t2, venue, stage, t1_wr, t2_wr,
                              model, le_t1, le_t2, le_venue)
    proba   = model.predict_proba(feats)[0]
    classes = list(model.classes_)
    p_t1    = float(proba[classes.index(1)])
    return {
        "winner": t1 if p_t1 >= 0.5 else t2,
        "p_t1":   round(p_t1, 4),
        "p_t2":   round(1 - p_t1, 4),
    }

# ── Simulation ────────────────────────────────────────────────────────────────
def win_rates_from_points(pts):
    return {
        t: round(v["won"] / v["played"], 4) if v.get("played", 0) > 0 else 0.5
        for t, v in pts.items()
    }

def simulate(points_table, fixtures, n=1000):
    model, le_t1, le_t2, le_venue = load_artifacts()
    wr = win_rates_from_points(points_table)
    champion_count = defaultdict(int)
    playoff_count  = defaultdict(int)

    # Pre-compute probabilities for breakdown
    breakdown = []
    for fix in fixtures:
        t1, t2  = fix["team1"], fix["team2"]
        venue   = fix.get("venue", "")
        stage   = fix.get("stage", "league")
        feats   = build_features(t1, t2, venue, stage,
                                  wr.get(t1, 0.5), wr.get(t2, 0.5),
                                  model, le_t1, le_t2, le_venue)
        proba   = model.predict_proba(feats)[0]
        p_t1    = float(proba[list(model.classes_).index(1)])
        breakdown.append({
            "match_id":      fix.get("match_id", f"{t1} vs {t2}"),
            "team1": t1, "team2": t2,
            "p_t1":  round(p_t1, 4),
            "p_t2":  round(1 - p_t1, 4),
            "likely_winner": t1 if p_t1 >= 0.5 else t2,
        })

    for _ in range(n):
        pts = deepcopy(points_table)
        for i, fix in enumerate(fixtures):
            t1, t2  = fix["team1"], fix["team2"]
            p_t1    = breakdown[i]["p_t1"]
            winner  = t1 if random.random() < p_t1 else t2
            loser   = t2 if winner == t1 else t1
            for team in [winner, loser]:
                if team not in pts:
                    pts[team] = {"played":0,"won":0,"lost":0,"pts":0,"nrr":0.0}
            pts[winner]["pts"] = pts[winner].get("pts",0) + 2
            pts[winner]["won"] = pts[winner].get("won",0) + 1
            pts[loser]["lost"] = pts[loser].get("lost",0) + 1

        ranked = sorted(pts.keys(),
                        key=lambda t: (pts[t].get("pts",0), pts[t].get("nrr",0.0)),
                        reverse=True)
        top4 = ranked[:4]
        for t in top4:
            playoff_count[t] += 1

        def win_p(a, b):
            feats = build_features(a, b, "", "qualifier",
                                    wr.get(a,0.5), wr.get(b,0.5),
                                    model, le_t1, le_t2, le_venue)
            p = float(model.predict_proba(feats)[0][list(model.classes_).index(1)])
            return a if random.random() < p else b

        if len(top4) >= 4:
            q1w  = win_p(top4[0], top4[1])
            q1l  = top4[1] if q1w == top4[0] else top4[0]
            ew   = win_p(top4[2], top4[3])
            q2w  = win_p(q1l, ew)
            champ = win_p(q1w, q2w)
            champion_count[champ] += 1

    teams = list(points_table.keys())
    return {
        "champion_probability":  {t: round(champion_count[t]/n*100,1) for t in teams},
        "playoff_qualification": {t: round(playoff_count[t]/n*100,1)  for t in teams},
        "match_breakdown": breakdown,
        "simulations_run": n,
    }

# ── Accuracy loader ───────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_accuracy():
    p = Path(__file__).parent / "data" / "accuracy_log.json"
    if not p.exists():
        return []
    with open(p) as f:
        return json.load(f)

def save_accuracy(records):
    p = Path(__file__).parent / "data" / "accuracy_log.json"
    with open(p, "w") as f:
        json.dump(records, f, indent=2)
    load_accuracy.clear()

# ─────────────────────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────────────────────

teams      = load_json("ipl_teams.json") or []
points_raw = load_json("ipl_points_table.json") or {}
fixtures   = load_json("ipl_fixtures.json") or []
venues     = sorted(set(f.get("venue","") for f in fixtures if f.get("venue")))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">TATA IPL 2026 · AI</div>
  <h1 class="hero-title">Match Predictor</h1>
  <p class="hero-sub">Random Forest · Trained on 17 seasons of IPL data</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_sim, tab_accuracy, tab_admin = st.tabs([
    "🏏 Predict", "📊 Simulate Season", "✅ Accuracy", "🔧 Admin"
])

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-title">Single Match Prediction</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 2])

    with col1:
        st.markdown('<div class="team-label">TEAM 1</div>', unsafe_allow_html=True)
        t1 = st.selectbox("Team 1", teams, key="t1", label_visibility="collapsed")
        t1_wr_pct = st.slider("Team 1 recent win rate %", 0, 100, 50, key="t1wr")

    with col2:
        st.markdown('<div class="vs-block">VS</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="team-label">TEAM 2</div>', unsafe_allow_html=True)
        t2 = st.selectbox("Team 2", [t for t in teams if t != t1], key="t2", label_visibility="collapsed")
        t2_wr_pct = st.slider("Team 2 recent win rate %", 0, 100, 50, key="t2wr")

    col_v, col_s = st.columns(2)
    with col_v:
        venue = st.selectbox("Venue", ["(Neutral)"] + venues, key="venue")
        venue = "" if venue == "(Neutral)" else venue
    with col_s:
        stage = st.selectbox("Stage", ["league", "qualifier", "eliminator", "final"], key="stage")

    if st.button("🔮 Predict Winner", use_container_width=True, type="primary"):
        if t1 == t2:
            st.error("Please select two different teams.")
        else:
            result = predict_match(t1, t2, venue, stage, t1_wr_pct/100, t2_wr_pct/100)
            winner = result["winner"]
            p1     = result["p_t1"] * 100
            p2     = result["p_t2"] * 100

            st.markdown(f"""
            <div class="result-card">
              <div class="result-winner">🏆 {winner}</div>
              <div class="result-sub">predicted to win</div>
              <div class="prob-row">
                <div class="prob-item">
                  <div class="prob-name">{t1}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{p1:.0f}%"></div>
                  </div>
                  <div class="prob-pct">{p1:.1f}%</div>
                </div>
                <div class="prob-item">
                  <div class="prob-name">{t2}</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar t2" style="width:{p2:.0f}%"></div>
                  </div>
                  <div class="prob-pct">{p2:.1f}%</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Save to session for accuracy logging
            st.session_state["last_pred"] = {
                "match_id":  f"IPL_{t1.replace(' ','_')}_{t2.replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M')}",
                "team1": t1, "team2": t2, "venue": venue, "stage": stage,
                "predicted_winner": winner,
                "p_t1": round(p1, 1), "p_t2": round(p2, 1),
                "actual_winner": None, "correct": None,
                "timestamp": datetime.now().isoformat(),
            }

    # Log actual result inline
    if "last_pred" in st.session_state and st.session_state["last_pred"]["actual_winner"] is None:
        pred = st.session_state["last_pred"]
        st.markdown("---")
        st.markdown("**📝 Log actual result for this prediction:**")
        actual = st.radio("Who actually won?",
                          [pred["team1"], pred["team2"], "Match not played yet"],
                          horizontal=True, key="actual_inline")
        if st.button("Save Result", key="save_inline"):
            if actual != "Match not played yet":
                records = load_accuracy()
                entry   = {**pred, "actual_winner": actual,
                           "correct": "yes" if actual == pred["predicted_winner"] else "no"}
                records.append(entry)
                save_accuracy(records)
                st.session_state["last_pred"]["actual_winner"] = actual
                st.success(f"✅ Saved! Prediction was {'correct ✓' if entry['correct']=='yes' else 'incorrect ✗'}")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — SIMULATE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown('<div class="section-title">Season Simulation</div>', unsafe_allow_html=True)

    if not fixtures:
        st.warning("No fixtures loaded. Upload `ipl_fixtures.json` to the `data/` folder.")
    elif not points_raw:
        st.warning("No points table loaded. Upload `ipl_points_table.json` to the `data/` folder.")
    else:
        remaining = [f for f in fixtures if not f.get("played", False)]
        st.info(f"**{len(remaining)}** remaining fixtures found. Running {1000:,} Monte Carlo simulations.")

        if st.button("🎲 Run Simulation", use_container_width=True, type="primary"):
            with st.spinner("Simulating season…"):
                result = simulate(points_raw, remaining, n=1000)

            st.markdown("### 🏆 Champion Probability")
            champ = dict(sorted(result["champion_probability"].items(),
                                key=lambda x: -x[1]))
            for team, pct in champ.items():
                if pct > 0:
                    st.markdown(f"""
                    <div class="sim-row">
                      <div class="sim-team">{team}</div>
                      <div class="sim-bar-wrap">
                        <div class="sim-bar" style="width:{min(pct*3,100):.0f}%"></div>
                      </div>
                      <div class="sim-pct">{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("### 📋 Match Breakdown")
            for m in result["match_breakdown"]:
                w = m["likely_winner"]
                col_a, col_b, col_c = st.columns([3, 1, 3])
                with col_a:
                    bold = "**" if w == m["team1"] else ""
                    st.markdown(f"{bold}{m['team1']}{bold} — {m['p_t1']*100:.0f}%")
                with col_b:
                    st.markdown("<div style='text-align:center;color:#888'>vs</div>",
                                unsafe_allow_html=True)
                with col_c:
                    bold = "**" if w == m["team2"] else ""
                    st.markdown(f"{bold}{m['team2']}{bold} — {m['p_t2']*100:.0f}%")

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — ACCURACY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_accuracy:
    st.markdown('<div class="section-title">Model Accuracy</div>', unsafe_allow_html=True)

    records = load_accuracy()
    resolved = [r for r in records if r.get("correct") in ("yes","no")]

    if not resolved:
        st.info("No resolved predictions yet. Make predictions and log actual results.")
    else:
        total   = len(resolved)
        correct = sum(1 for r in resolved if r["correct"] == "yes")
        acc_pct = correct / total * 100

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Predictions", total)
        c2.metric("Correct", correct)
        c3.metric("Incorrect", total - correct)
        c4.metric("Accuracy", f"{acc_pct:.1f}%",
                  delta=f"{acc_pct-50:.1f}% vs coin flip",
                  delta_color="normal")

        # Accuracy bar
        st.markdown(f"""
        <div class="acc-bar-wrap">
          <div class="acc-bar" style="width:{acc_pct:.1f}%">
            <span class="acc-label">{acc_pct:.1f}%</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Recent predictions table
        st.markdown("### Recent Predictions")
        recent = sorted(resolved, key=lambda x: x.get("timestamp",""), reverse=True)[:20]
        for r in recent:
            icon = "✅" if r["correct"] == "yes" else "❌"
            st.markdown(f"""
            <div class="acc-row {'correct' if r['correct']=='yes' else 'wrong'}">
              <span class="acc-icon">{icon}</span>
              <span class="acc-match">{r['team1']} vs {r['team2']}</span>
              <span class="acc-pred">Predicted: <b>{r['predicted_winner']}</b></span>
              <span class="acc-actual">Actual: <b>{r['actual_winner']}</b></span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ADMIN
# ═══════════════════════════════════════════════════════════════════════════════
with tab_admin:
    st.markdown('<div class="section-title">Admin Panel</div>', unsafe_allow_html=True)

    admin_key = st.text_input("Admin Password", type="password", key="admin_pw")
    ADMIN_PASS = os.environ.get("ADMIN_PASSWORD", "ipl2026admin")

    if admin_key != ADMIN_PASS:
        st.warning("Enter admin password to continue.")
        st.stop()

    st.success("✅ Authenticated")

    adm1, adm2 = st.tabs(["📅 Fixtures", "📊 Points Table"])

    with adm1:
        st.markdown("#### Upload / Edit Fixtures JSON")
        uploaded = st.file_uploader("Upload ipl_fixtures.json", type="json", key="fix_upload")
        if uploaded:
            data = json.load(uploaded)
            p = Path(__file__).parent / "data" / "ipl_fixtures.json"
            with open(p,"w") as f:
                json.dump(data, f, indent=2)
            st.success(f"✅ Saved {len(data)} fixtures")
            load_json.clear()

        st.markdown("#### Current Fixtures")
        if fixtures:
            for i, fix in enumerate(fixtures):
                cols = st.columns([3,1,3,2,2])
                cols[0].write(fix.get("team1",""))
                cols[1].write("vs")
                cols[2].write(fix.get("team2",""))
                cols[3].write(fix.get("stage","league"))
                played = cols[4].checkbox("Played", value=fix.get("played",False), key=f"played_{i}")
                if played != fix.get("played", False):
                    fixtures[i]["played"] = played
                    p = Path(__file__).parent / "data" / "ipl_fixtures.json"
                    with open(p,"w") as f:
                        json.dump(fixtures, f, indent=2)
                    load_json.clear()
        else:
            st.info("No fixtures loaded.")

    with adm2:
        st.markdown("#### Upload / Edit Points Table JSON")
        uploaded2 = st.file_uploader("Upload ipl_points_table.json", type="json", key="pts_upload")
        if uploaded2:
            data2 = json.load(uploaded2)
            p = Path(__file__).parent / "data" / "ipl_points_table.json"
            with open(p,"w") as f:
                json.dump(data2, f, indent=2)
            st.success("✅ Points table saved")
            load_json.clear()

        st.markdown("#### Edit Points Manually")
        if points_raw:
            updated = deepcopy(points_raw)
            for team in list(points_raw.keys()):
                row = points_raw[team]
                with st.expander(team):
                    c1,c2,c3,c4 = st.columns(4)
                    updated[team]["played"] = c1.number_input("Played", value=row.get("played",0), min_value=0, key=f"p_{team}")
                    updated[team]["won"]    = c2.number_input("Won",    value=row.get("won",0),    min_value=0, key=f"w_{team}")
                    updated[team]["lost"]   = c3.number_input("Lost",   value=row.get("lost",0),   min_value=0, key=f"l_{team}")
                    updated[team]["nrr"]    = c4.number_input("NRR",    value=float(row.get("nrr",0.0)), key=f"n_{team}", format="%.3f")
                    updated[team]["pts"]    = updated[team]["won"] * 2
            if st.button("💾 Save Points Table", type="primary"):
                p = Path(__file__).parent / "data" / "ipl_points_table.json"
                with open(p,"w") as f:
                    json.dump(updated, f, indent=2)
                st.success("✅ Saved!")
                load_json.clear()
