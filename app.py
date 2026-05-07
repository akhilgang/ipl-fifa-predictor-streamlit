"""
app.py  — IPL 2026 + FIFA World Cup 2026 · AI Predictor · Streamlit
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
    page_title="Sport AI · Predictor 2026",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load custom CSS ───────────────────────────────────────────────────────────
with open(Path(__file__).parent / "style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  IPL ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ipl_artifacts():
    base = Path(__file__).parent / "models"
    model    = joblib.load(base / "ipl_model.pkl")
    le_t1    = joblib.load(base / "ipl_le_t1.pkl")
    le_t2    = joblib.load(base / "ipl_le_t2.pkl")
    le_venue = joblib.load(base / "ipl_le_venue.pkl")
    return model, le_t1, le_t2, le_venue

@st.cache_resource
def load_fifa_artifacts():
    base = Path(__file__).parent / "models"
    model    = joblib.load(base / "fifa_model.pkl")
    le_home  = joblib.load(base / "fifa_le_home.pkl")
    le_away  = joblib.load(base / "fifa_le_away.pkl")
    return model, le_home, le_away

@st.cache_data
def load_json(filename):
    p = Path(__file__).parent / "data" / filename
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

# ─────────────────────────────────────────────────────────────────────────────
#  IPL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _enc(le, val, fallback=0):
    try:    return int(le.transform([val])[0])
    except: return fallback

IPL_STAGE_W = {"league": 1, "qualifier": 2, "eliminator": 2, "final": 3}

def build_ipl_features(t1, t2, venue, stage, t1_wr, t2_wr, model, le_t1, le_t2, le_venue):
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

def predict_ipl(t1, t2, venue, stage, t1_wr, t2_wr):
    model, le_t1, le_t2, le_venue = load_ipl_artifacts()
    feats = build_ipl_features(t1, t2, venue, stage, t1_wr, t2_wr,
                               model, le_t1, le_t2, le_venue)
    proba   = model.predict_proba(feats)[0]
    classes = list(model.classes_)
    p_t1    = float(proba[classes.index(1)])
    return {"winner": t1 if p_t1 >= 0.5 else t2,
            "p_t1": round(p_t1, 4), "p_t2": round(1 - p_t1, 4)}

def win_rates_from_points(pts):
    return {
        t: round(v["won"] / v["played"], 4) if v.get("played", 0) > 0 else 0.5
        for t, v in pts.items()
    }

def simulate_ipl(points_table, fixtures, n=1000):
    model, le_t1, le_t2, le_venue = load_ipl_artifacts()
    wr = win_rates_from_points(points_table)
    champion_count = defaultdict(int)
    playoff_count  = defaultdict(int)
    breakdown = []
    for fix in fixtures:
        t1, t2 = fix["team1"], fix["team2"]
        venue  = fix.get("venue", "")
        stage  = fix.get("stage", "league")
        feats  = build_ipl_features(t1, t2, venue, stage,
                                    wr.get(t1, 0.5), wr.get(t2, 0.5),
                                    model, le_t1, le_t2, le_venue)
        proba  = model.predict_proba(feats)[0]
        p_t1   = float(proba[list(model.classes_).index(1)])
        breakdown.append({
            "match_id": fix.get("match_id", f"{t1} vs {t2}"),
            "team1": t1, "team2": t2,
            "p_t1": round(p_t1, 4), "p_t2": round(1 - p_t1, 4),
            "likely_winner": t1 if p_t1 >= 0.5 else t2,
        })
    for _ in range(n):
        pts = deepcopy(points_table)
        for i, fix in enumerate(fixtures):
            t1, t2 = fix["team1"], fix["team2"]
            p_t1   = breakdown[i]["p_t1"]
            winner = t1 if random.random() < p_t1 else t2
            loser  = t2 if winner == t1 else t1
            for team in [winner, loser]:
                if team not in pts:
                    pts[team] = {"played": 0, "won": 0, "lost": 0, "pts": 0, "nrr": 0.0}
            pts[winner]["pts"] = pts[winner].get("pts", 0) + 2
            pts[winner]["won"] = pts[winner].get("won", 0) + 1
            pts[loser]["lost"] = pts[loser].get("lost", 0) + 1
        ranked = sorted(pts.keys(), key=lambda t: (pts[t].get("pts", 0), pts[t].get("nrr", 0.0)), reverse=True)
        top4 = ranked[:4]
        for t in top4:
            playoff_count[t] += 1
        def win_p(a, b):
            feats = build_ipl_features(a, b, "", "qualifier",
                                       wr.get(a, 0.5), wr.get(b, 0.5),
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
        "champion_probability":  {t: round(champion_count[t]/n*100, 1) for t in teams},
        "playoff_qualification": {t: round(playoff_count[t]/n*100, 1)  for t in teams},
        "match_breakdown": breakdown,
        "simulations_run": n,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  FIFA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

# Team aliases: FIFA 2026 official names → training data names
TEAM_ALIASES = {
    "Korea Republic":     "South Korea",
    "Czechia":            "Czech Republic",
    "Bosnia Herzegovina": "Bosnia and Herzegovina",
    "Cote dIvoire":       "Ivory Coast",
    "Congo DR":           "DR Congo",
    "IR Iran":            "Iran",
    "Cabo Verde":         "Cape Verde",
    "Curacao":            "Curaçao",
    "USA":                "United States",
}

FIFA_GROUPS = {
    "A": ["Mexico",       "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada",       "Bosnia Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil",       "Morocco",      "Haiti",          "Scotland"],
    "D": ["USA",          "Paraguay",     "Australia",      "Turkey"],
    "E": ["Germany",      "Curacao",      "Cote dIvoire",   "Ecuador"],
    "F": ["Netherlands",  "Japan",        "Sweden",         "Tunisia"],
    "G": ["Belgium",      "Egypt",        "IR Iran",        "New Zealand"],
    "H": ["Spain",        "Cabo Verde",   "Saudi Arabia",   "Uruguay"],
    "I": ["France",       "Senegal",      "Iraq",           "Norway"],
    "J": ["Argentina",    "Algeria",      "Austria",        "Jordan"],
    "K": ["Portugal",     "Congo DR",     "Uzbekistan",     "Colombia"],
    "L": ["England",      "Croatia",      "Ghana",          "Panama"],
}

FIFA_GROUP_FIXTURES = [
    # Group A
    ("Mexico", "South Africa", "2026-06-11"),
    ("Korea Republic", "Czechia", "2026-06-12"),
    ("Mexico", "Korea Republic", "2026-06-19"),
    ("Czechia", "South Africa", "2026-06-18"),
    ("Czechia", "Mexico", "2026-06-22"),
    ("South Africa", "Korea Republic", "2026-06-22"),
    # Group B
    ("Canada", "Bosnia Herzegovina", "2026-06-12"),
    ("Qatar", "Switzerland", "2026-06-13"),
    ("Canada", "Qatar", "2026-06-18"),
    ("Bosnia Herzegovina", "Switzerland", "2026-06-18"),
    ("Switzerland", "Canada", "2026-06-22"),
    ("Bosnia Herzegovina", "Qatar", "2026-06-22"),
    # Group C
    ("Brazil", "Morocco", "2026-06-13"),
    ("Haiti", "Scotland", "2026-06-14"),
    ("Brazil", "Haiti", "2026-06-19"),
    ("Scotland", "Morocco", "2026-06-19"),
    ("Morocco", "Haiti", "2026-06-22"),
    ("Scotland", "Brazil", "2026-06-22"),
    # Group D
    ("USA", "Paraguay", "2026-06-13"),
    ("Turkey", "USA", "2026-06-14"),
    ("Australia", "Turkey", "2026-06-19"),
    ("USA", "Australia", "2026-06-20"),
    ("Paraguay", "Turkey", "2026-06-25"),
    ("Paraguay", "Australia", "2026-06-25"),
    # Group E
    ("Germany", "Curacao", "2026-06-14"),
    ("Cote dIvoire", "Ecuador", "2026-06-14"),
    ("Germany", "Cote dIvoire", "2026-06-20"),
    ("Ecuador", "Curacao", "2026-06-20"),
    ("Ecuador", "Germany", "2026-06-23"),
    ("Curacao", "Cote dIvoire", "2026-06-24"),
    # Group F
    ("Netherlands", "Japan", "2026-06-15"),
    ("Sweden", "Tunisia", "2026-06-15"),
    ("Netherlands", "Sweden", "2026-06-20"),
    ("Tunisia", "Japan", "2026-06-20"),
    ("Tunisia", "Netherlands", "2026-06-24"),
    ("Japan", "Sweden", "2026-06-25"),
    # Group G
    ("Belgium", "Egypt", "2026-06-16"),
    ("IR Iran", "New Zealand", "2026-06-16"),
    ("Belgium", "New Zealand", "2026-06-25"),
    ("Egypt", "IR Iran", "2026-06-26"),
    ("New Zealand", "Egypt", "2026-06-21"),
    ("Belgium", "IR Iran", "2026-06-21"),
    # Group H
    ("Spain", "Cabo Verde", "2026-06-16"),
    ("Saudi Arabia", "Uruguay", "2026-06-15"),
    ("Spain", "Saudi Arabia", "2026-06-21"),
    ("Uruguay", "Cabo Verde", "2026-06-22"),
    ("Cabo Verde", "Saudi Arabia", "2026-06-25"),
    ("Uruguay", "Spain", "2026-06-25"),
    # Group I
    ("France", "Senegal", "2026-06-17"),
    ("Iraq", "Norway", "2026-06-16"),
    ("Norway", "Senegal", "2026-06-21"),
    ("France", "Iraq", "2026-06-21"),
    ("Norway", "France", "2026-06-25"),
    ("Senegal", "Iraq", "2026-06-25"),
    # Group J
    ("Argentina", "Algeria", "2026-06-17"),
    ("Austria", "Jordan", "2026-06-16"),
    ("Argentina", "Austria", "2026-06-21"),
    ("Jordan", "Algeria", "2026-06-21"),
    ("Jordan", "Argentina", "2026-06-26"),
    ("Algeria", "Austria", "2026-06-26"),
    # Group K
    ("Portugal", "Congo DR", "2026-06-18"),
    ("Uzbekistan", "Colombia", "2026-06-17"),
    ("Portugal", "Uzbekistan", "2026-06-23"),
    ("Colombia", "Congo DR", "2026-06-22"),
    ("Colombia", "Portugal", "2026-06-26"),
    ("Congo DR", "Uzbekistan", "2026-06-26"),
    # Group L
    ("England", "Croatia", "2026-06-17"),
    ("Ghana", "Panama", "2026-06-16"),
    ("England", "Ghana", "2026-06-21"),
    ("Panama", "Croatia", "2026-06-22"),
    ("Panama", "England", "2026-06-25"),
    ("Croatia", "Ghana", "2026-06-25"),
]

TOURNAMENT_WEIGHT_FIFA = 10  # FIFA World Cup

def resolve_team(name):
    return TEAM_ALIASES.get(name, name)

def _enc_fifa(le, val, fallback=0):
    try:    return int(le.transform([val])[0])
    except: return fallback

def predict_fifa(home, away, h_wr=0.5, a_wr=0.5):
    """
    Returns {"win": p, "draw": p, "loss": p, "likely": outcome}
    outcome is from home team perspective: win/draw/loss
    """
    model, le_home, le_away = load_fifa_artifacts()
    h_res = resolve_team(home)
    a_res = resolve_team(away)
    h_enc = _enc_fifa(le_home, h_res)
    a_enc = _enc_fifa(le_away, a_res)
    wr_diff = h_wr - a_wr
    gd_diff = 0.0
    feats = [[
        h_enc, a_enc,
        TOURNAMENT_WEIGHT_FIFA,
        1,            # is_neutral (World Cup = neutral venue)
        h_wr, a_wr,
        0.0, 0.0,     # avg_gd placeholders
        wr_diff, gd_diff,
        0.33,         # h2h default
    ]]
    proba   = model.predict_proba(feats)[0]
    classes = list(model.classes_)
    prob_map = {c: float(proba[i]) for i, c in enumerate(classes)}
    p_win  = prob_map.get("win",  0.33)
    p_draw = prob_map.get("draw", 0.34)
    p_loss = prob_map.get("loss", 0.33)
    # Normalise
    total  = p_win + p_draw + p_loss
    p_win, p_draw, p_loss = p_win/total, p_draw/total, p_loss/total
    likely = "win" if p_win >= p_draw and p_win >= p_loss else (
             "draw" if p_draw >= p_loss else "loss")
    return {
        "win": round(p_win, 4), "draw": round(p_draw, 4), "loss": round(p_loss, 4),
        "likely": likely,
    }

def simulate_group(group_teams):
    """Simulate a single group, return standings dict {team: {pts, gd, won, drawn, lost}}"""
    standings = {t: {"pts": 0, "gd": 0, "won": 0, "drawn": 0, "lost": 0, "played": 0}
                 for t in group_teams}
    pairs = [(group_teams[i], group_teams[j])
             for i in range(len(group_teams)) for j in range(i+1, len(group_teams))]
    for home, away in pairs:
        res = predict_fifa(home, away)
        r   = random.random()
        if r < res["win"]:
            outcome = "win"
        elif r < res["win"] + res["draw"]:
            outcome = "draw"
        else:
            outcome = "loss"
        standings[home]["played"] += 1
        standings[away]["played"] += 1
        if outcome == "win":
            standings[home]["pts"] += 3; standings[home]["won"] += 1
            standings[away]["lost"] += 1; standings[home]["gd"] += 1; standings[away]["gd"] -= 1
        elif outcome == "draw":
            standings[home]["pts"] += 1; standings[home]["drawn"] += 1
            standings[away]["pts"] += 1; standings[away]["drawn"] += 1
        else:
            standings[away]["pts"] += 3; standings[away]["won"] += 1
            standings[home]["lost"] += 1; standings[away]["gd"] += 1; standings[home]["gd"] -= 1
    ranked = sorted(standings.keys(),
                    key=lambda t: (standings[t]["pts"], standings[t]["gd"]), reverse=True)
    return ranked, standings

def simulate_knockout_match(t1, t2):
    res = predict_fifa(t1, t2)
    r   = random.random()
    # In knockouts no draw — use win probability, else coin-flip weighted by loss
    # If draw: 50-50 extra time
    if r < res["win"]:
        return t1
    elif r < res["win"] + res["draw"]:
        return t1 if random.random() < 0.5 else t2
    else:
        return t2

def simulate_tournament(n=500):
    """Monte-Carlo tournament simulation, returns champion & finalist counts."""
    champion_count = defaultdict(int)
    finalist_count = defaultdict(int)
    sf_count       = defaultdict(int)
    qf_count       = defaultdict(int)

    for _ in range(n):
        group_winners  = []
        group_runners  = []
        third_place    = []

        for grp, teams in FIFA_GROUPS.items():
            ranked, _ = simulate_group(list(teams))
            group_winners.append(ranked[0])
            group_runners.append(ranked[1])
            third_place.append(ranked[2])

        # Best 4 third-place teams advance
        third_place_sorted = sorted(
            third_place,
            key=lambda t: predict_fifa(t, "France")["win"],
            reverse=True
        )[:4]

        ro32 = group_winners + group_runners + third_place_sorted
        random.shuffle(ro32)  # simplified bracket

        def run_round(teams):
            winners = []
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    winners.append(simulate_knockout_match(teams[i], teams[i+1]))
                else:
                    winners.append(teams[i])
            return winners

        qf_teams = run_round(ro32)
        for t in qf_teams:
            qf_count[t] += 1

        sf_teams = run_round(qf_teams)
        for t in sf_teams:
            sf_count[t] += 1

        f_teams = run_round(sf_teams)
        for t in f_teams:
            finalist_count[t] += 1

        champ = run_round(f_teams)
        if champ:
            champion_count[champ[0]] += 1

    all_teams = [t for teams in FIFA_GROUPS.values() for t in teams]
    return {
        "champion":  {t: round(champion_count[t]/n*100, 1) for t in all_teams},
        "finalist":  {t: round(finalist_count[t]/n*100, 1) for t in all_teams},
        "semifinal": {t: round(sf_count[t]/n*100, 1)       for t in all_teams},
        "quarterfinal": {t: round(qf_count[t]/n*100, 1)    for t in all_teams},
        "n": n,
    }

# ─────────────────────────────────────────────────────────────────────────────
#  ACCURACY
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def load_accuracy(sport="ipl"):
    fname = f"{sport}_accuracy_log.json"
    p = Path(__file__).parent / "data" / fname
    if not p.exists():
        if sport == "ipl":
            p = Path(__file__).parent / "data" / "accuracy_log.json"
    if not p.exists():
        return []
    # ← add this guard
    if p.stat().st_size == 0:
        return []
    with open(p) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []   # corrupt file → treat as empty

def save_accuracy(records, sport="ipl"):
    fname = f"{sport}_accuracy_log.json"
    p = Path(__file__).parent / "data" / fname
    with open(p, "w") as f:
        json.dump(records, f, indent=2)
    load_accuracy.clear()

# ─────────────────────────────────────────────────────────────────────────────
#  SHARED UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_accuracy_tab(sport="ipl"):
    records  = load_accuracy(sport)
    resolved = [r for r in records if r.get("correct") in ("yes", "no")]
    if not resolved:
        st.info("No resolved predictions yet. Make predictions and log actual results.")
        return
    total   = len(resolved)
    correct = sum(1 for r in resolved if r["correct"] == "yes")
    acc_pct = correct / total * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Predictions", total)
    c2.metric("Correct", correct)
    c3.metric("Incorrect", total - correct)
    c4.metric("Accuracy", f"{acc_pct:.1f}%",
              delta=f"{acc_pct-50:.1f}% vs coin flip", delta_color="normal")
    st.markdown(f"""
    <div class="acc-bar-wrap">
      <div class="acc-bar" style="width:{acc_pct:.1f}%">
        <span class="acc-label">{acc_pct:.1f}%</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    recent = sorted(resolved, key=lambda x: x.get("timestamp", ""), reverse=True)[:20]
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

def render_prob_bars(t1, t2, p1_pct, p2_pct, winner):
    st.markdown(f"""
    <div class="result-card">
      <div class="result-winner">🏆 {winner}</div>
      <div class="result-sub">predicted to win</div>
      <div class="prob-row">
        <div class="prob-item">
          <div class="prob-name">{t1}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar" style="width:{p1_pct:.0f}%"></div>
          </div>
          <div class="prob-pct">{p1_pct:.1f}%</div>
        </div>
        <div class="prob-item">
          <div class="prob-name">{t2}</div>
          <div class="prob-bar-wrap">
            <div class="prob-bar t2" style="width:{p2_pct:.0f}%"></div>
          </div>
          <div class="prob-pct">{p2_pct:.1f}%</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADS
# ─────────────────────────────────────────────────────────────────────────────

ipl_teams      = load_json("ipl_teams.json") or []
ipl_points_raw = load_json("ipl_points_table.json") or {}
ipl_fixtures   = load_json("ipl_fixtures.json") or []
ipl_venues     = sorted(set(f.get("venue", "") for f in ipl_fixtures if f.get("venue")))

fifa_teams_raw = load_json("fifa_teams.json") or []
# All FIFA 2026 participating teams
FIFA_2026_TEAMS = sorted(set(t for teams in FIFA_GROUPS.values() for t in teams))

# ─────────────────────────────────────────────────────────────────────────────
#  HERO + SPORT SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
  <div class="hero-badge">AI · SPORTS PREDICTOR 2026</div>
  <h1 class="hero-title">Sports Predictor</h1>
  <p class="hero-sub">Random Forest · IPL 2026 + FIFA World Cup 2026</p>
</div>
""", unsafe_allow_html=True)

# Sport selector — centered
st.markdown('<div class="sport-selector-wrap">', unsafe_allow_html=True)
sport = st.radio(
    "Select Sport",
    ["🏏 IPL 2026", "⚽ FIFA World Cup 2026"],
    horizontal=True,
    label_visibility="collapsed",
    key="sport_radio",
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  IPL SECTION
# ═══════════════════════════════════════════════════════════════════════════════

if sport == "🏏 IPL 2026":

    tab_predict, tab_sim, tab_accuracy, tab_admin = st.tabs([
        "🏏 Predict", "📊 Simulate Season", "✅ Accuracy", "🔧 Admin"
    ])

    # ── IPL TAB 1 — PREDICT ──────────────────────────────────────────────────
    with tab_predict:
        st.markdown('<div class="section-title">Single Match Prediction</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown('<div class="team-label">TEAM 1</div>', unsafe_allow_html=True)
            t1 = st.selectbox("Team 1", ipl_teams, key="ipl_t1", label_visibility="collapsed")
            t1_wr_pct = st.slider("Team 1 recent win rate %", 0, 100, 50, key="ipl_t1wr")
        with col2:
            st.markdown('<div class="vs-block">VS</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="team-label">TEAM 2</div>', unsafe_allow_html=True)
            t2 = st.selectbox("Team 2", [t for t in ipl_teams if t != t1], key="ipl_t2", label_visibility="collapsed")
            t2_wr_pct = st.slider("Team 2 recent win rate %", 0, 100, 50, key="ipl_t2wr")

        col_v, col_s = st.columns(2)
        with col_v:
            venue = st.selectbox("Venue", ["(Neutral)"] + ipl_venues, key="ipl_venue")
            venue = "" if venue == "(Neutral)" else venue
        with col_s:
            stage = st.selectbox("Stage", ["league", "qualifier", "eliminator", "final"], key="ipl_stage")

        if st.button("🔮 Predict Winner", use_container_width=True, type="primary", key="ipl_predict_btn"):
            if t1 == t2:
                st.error("Please select two different teams.")
            else:
                result = predict_ipl(t1, t2, venue, stage, t1_wr_pct/100, t2_wr_pct/100)
                winner = result["winner"]
                p1     = result["p_t1"] * 100
                p2     = result["p_t2"] * 100
                render_prob_bars(t1, t2, p1, p2, winner)
                st.session_state["ipl_last_pred"] = {
                    "match_id":  f"IPL_{t1.replace(' ','_')}_{t2.replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M')}",
                    "team1": t1, "team2": t2, "venue": venue, "stage": stage,
                    "predicted_winner": winner,
                    "p_t1": round(p1, 1), "p_t2": round(p2, 1),
                    "actual_winner": None, "correct": None,
                    "timestamp": datetime.now().isoformat(),
                }

    # ── IPL TAB 2 — SIMULATE ─────────────────────────────────────────────────
    with tab_sim:
        st.markdown('<div class="section-title">Season Simulation</div>', unsafe_allow_html=True)
        if not ipl_fixtures:
            st.warning("No fixtures loaded. Upload `ipl_fixtures.json` to the `data/` folder.")
        elif not ipl_points_raw:
            st.warning("No points table loaded. Upload `ipl_points_table.json` to the `data/` folder.")
        else:
            remaining = [f for f in ipl_fixtures if not f.get("played", False)]
            st.info(f"**{len(remaining)}** remaining fixtures found. Running 1,000 Monte Carlo simulations.")
            if st.button("🎲 Run Simulation", use_container_width=True, type="primary", key="ipl_sim_btn"):
                with st.spinner("Simulating season…"):
                    st.session_state["ipl_sim_result"] = simulate_ipl(ipl_points_raw, remaining, n=1000)

            if "ipl_sim_result" in st.session_state:
                result = st.session_state["ipl_sim_result"]
                st.markdown("### 🏆 Champion Probability")
                champ = dict(sorted(result["champion_probability"].items(), key=lambda x: -x[1]))
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
                        st.markdown("<div style='text-align:center;color:#888'>vs</div>", unsafe_allow_html=True)
                    with col_c:
                        bold = "**" if w == m["team2"] else ""
                        st.markdown(f"{bold}{m['team2']}{bold} — {m['p_t2']*100:.0f}%")

    # ── IPL TAB 3 — ACCURACY ─────────────────────────────────────────────────
    with tab_accuracy:
        st.markdown('<div class="section-title">Model Accuracy</div>', unsafe_allow_html=True)
        render_accuracy_tab("ipl")

    # ── IPL TAB 4 — ADMIN ────────────────────────────────────────────────────
    with tab_admin:
        st.markdown('<div class="section-title">Admin Panel</div>', unsafe_allow_html=True)
        admin_key = st.text_input("Admin Password", type="password", key="ipl_admin_pw")
        ADMIN_PASS = os.environ.get("ADMIN_PASSWORD", "ipl2026admin")
        if admin_key != ADMIN_PASS:
            st.warning("Enter admin password to continue.")
            st.stop()
        st.success("✅ Authenticated")
        adm1, adm2 = st.tabs(["📅 Fixtures", "📊 Points Table"])
        with adm1:
            st.markdown("#### Upload / Edit Fixtures JSON")
            uploaded = st.file_uploader("Upload ipl_fixtures.json", type="json", key="ipl_fix_upload")
            if uploaded:
                data = json.load(uploaded)
                p = Path(__file__).parent / "data" / "ipl_fixtures.json"
                with open(p, "w") as f:
                    json.dump(data, f, indent=2)
                st.success(f"✅ Saved {len(data)} fixtures")
                load_json.clear()
            st.markdown("#### Current Fixtures")
            if ipl_fixtures:
                for i, fix in enumerate(ipl_fixtures):
                    cols = st.columns([3, 1, 3, 2, 2])
                    cols[0].write(fix.get("team1", ""))
                    cols[1].write("vs")
                    cols[2].write(fix.get("team2", ""))
                    cols[3].write(fix.get("stage", "league"))
                    played = cols[4].checkbox("Played", value=fix.get("played", False), key=f"ipl_played_{i}")
                    if played != fix.get("played", False):
                        ipl_fixtures[i]["played"] = played
                        p = Path(__file__).parent / "data" / "ipl_fixtures.json"
                        with open(p, "w") as f:
                            json.dump(ipl_fixtures, f, indent=2)
                        load_json.clear()
            else:
                st.info("No fixtures loaded.")
        with adm2:
            st.markdown("#### Upload / Edit Points Table JSON")
            uploaded2 = st.file_uploader("Upload ipl_points_table.json", type="json", key="ipl_pts_upload")
            if uploaded2:
                data2 = json.load(uploaded2)
                p = Path(__file__).parent / "data" / "ipl_points_table.json"
                with open(p, "w") as f:
                    json.dump(data2, f, indent=2)
                st.success("✅ Points table saved")
                load_json.clear()
            st.markdown("#### Edit Points Manually")
            if ipl_points_raw:
                updated = deepcopy(ipl_points_raw)
                for team in list(ipl_points_raw.keys()):
                    row = ipl_points_raw[team]
                    with st.expander(team):
                        c1, c2, c3, c4 = st.columns(4)
                        updated[team]["played"] = c1.number_input("Played", value=row.get("played", 0), min_value=0, key=f"ipl_p_{team}")
                        updated[team]["won"]    = c2.number_input("Won",    value=row.get("won", 0),    min_value=0, key=f"ipl_w_{team}")
                        updated[team]["lost"]   = c3.number_input("Lost",   value=row.get("lost", 0),   min_value=0, key=f"ipl_l_{team}")
                        updated[team]["nrr"]    = c4.number_input("NRR",    value=float(row.get("nrr", 0.0)), key=f"ipl_n_{team}", format="%.3f")
                        updated[team]["pts"]    = updated[team]["won"] * 2
                if st.button("💾 Save Points Table", type="primary", key="ipl_save_pts"):
                    p = Path(__file__).parent / "data" / "ipl_points_table.json"
                    with open(p, "w") as f:
                        json.dump(updated, f, indent=2)
                    st.success("✅ Saved!")
                    load_json.clear()

# ═══════════════════════════════════════════════════════════════════════════════
#  FIFA SECTION
# ═══════════════════════════════════════════════════════════════════════════════

else:  # FIFA World Cup 2026

    tab_predict_f, tab_groups, tab_sim_f, tab_fixtures_f, tab_acc_f = st.tabs([
        "⚽ Predict Match", "🗂️ Group Stage", "🌍 Tournament Sim", "📅 All Fixtures", "✅ Accuracy"
    ])

    # ── FIFA TAB 1 — PREDICT ─────────────────────────────────────────────────
    with tab_predict_f:
        st.markdown('<div class="section-title">Match Prediction</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1.5rem;">
        Trained on 35+ years of international football (1990–2025). 
        Predicts win / draw / loss from the home team's perspective.
        In World Cup knockout rounds, draw = extra time / penalties (50-50).
        </p>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 1, 2])
        with col1:
            st.markdown('<div class="team-label">HOME / TEAM 1</div>', unsafe_allow_html=True)
            f_t1 = st.selectbox("Home Team", FIFA_2026_TEAMS, key="fifa_t1", label_visibility="collapsed")
            f_t1_wr = st.slider("Team 1 recent win rate %", 0, 100, 50, key="fifa_t1wr") / 100

        with col2:
            st.markdown('<div class="vs-block">VS</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="team-label">AWAY / TEAM 2</div>', unsafe_allow_html=True)
            f_t2_opts = [t for t in FIFA_2026_TEAMS if t != f_t1]
            f_t2 = st.selectbox("Away Team", f_t2_opts, key="fifa_t2", label_visibility="collapsed")
            f_t2_wr = st.slider("Team 2 recent win rate %", 0, 100, 50, key="fifa_t2wr") / 100

        f_stage = st.selectbox(
            "Match Stage",
            ["Group Stage", "Round of 32", "Round of 16", "Quarter-final", "Semi-final", "Final"],
            key="fifa_stage"
        )

        if st.button("🔮 Predict Match", use_container_width=True, type="primary", key="fifa_predict_btn"):
            res = predict_fifa(f_t1, f_t2, f_t1_wr, f_t2_wr)
            p_win  = res["win"]  * 100
            p_draw = res["draw"] * 100
            p_loss = res["loss"] * 100
            likely = res["likely"]

            if likely == "win":
                winner_text = f_t1
                result_label = "WIN"
            elif likely == "loss":
                winner_text = f_t2
                result_label = "WIN"
            else:
                winner_text = "DRAW"
                result_label = ""

            st.markdown(f"""
            <div class="result-card">
              <div class="result-winner">{'🏆 ' + winner_text if result_label == 'WIN' else '🤝 ' + winner_text}</div>
              <div class="result-sub">{"predicted winner" if result_label == "WIN" else "most likely outcome"}</div>
              <div class="prob-row" style="margin-top:1.5rem;">
                <div class="prob-item">
                  <div class="prob-name">{f_t1} Win</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{p_win:.0f}%"></div>
                  </div>
                  <div class="prob-pct">{p_win:.1f}%</div>
                </div>
                <div class="prob-item">
                  <div class="prob-name">Draw</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar" style="width:{p_draw:.0f}%;background:linear-gradient(90deg,var(--gold),#d97706)"></div>
                  </div>
                  <div class="prob-pct">{p_draw:.1f}%</div>
                </div>
                <div class="prob-item">
                  <div class="prob-name">{f_t2} Win</div>
                  <div class="prob-bar-wrap">
                    <div class="prob-bar t2" style="width:{p_loss:.0f}%"></div>
                  </div>
                  <div class="prob-pct">{p_loss:.1f}%</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Determine predicted_winner for logging
            pw = f_t1 if likely == "win" else (f_t2 if likely == "loss" else "Draw")
            st.session_state["fifa_last_pred"] = {
                "match_id": f"FIFA_{f_t1.replace(' ','_')}_{f_t2.replace(' ','_')}_{datetime.now().strftime('%Y%m%d%H%M')}",
                "team1": f_t1, "team2": f_t2, "stage": f_stage,
                "predicted_winner": pw,
                "p_t1": round(p_win, 1), "p_t2": round(p_loss, 1), "p_draw": round(p_draw, 1),
                "actual_winner": None, "correct": None,
                "timestamp": datetime.now().isoformat(),
            }

    # ── FIFA TAB 2 — GROUP STAGE ──────────────────────────────────────────────
    with tab_groups:
        st.markdown('<div class="section-title">Group Stage Predictions</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1rem;">
        Predicted group standings based on model win probabilities (not simulated — deterministic best-guess).
        </p>
        """, unsafe_allow_html=True)

        selected_grp = st.selectbox(
            "Select Group",
            [f"Group {g}" for g in sorted(FIFA_GROUPS.keys())],
            key="fifa_grp_select"
        )
        grp_key = selected_grp.replace("Group ", "")
        grp_teams = FIFA_GROUPS[grp_key]

        # Build deterministic standings: run all pairs, award pts based on likely outcome
        standings = {t: {"pts": 0, "won": 0, "drawn": 0, "lost": 0} for t in grp_teams}
        pairs = [(grp_teams[i], grp_teams[j])
                 for i in range(len(grp_teams)) for j in range(i+1, len(grp_teams))]
        match_preds = []
        for home, away in pairs:
            res = predict_fifa(home, away)
            match_preds.append((home, away, res))
            likely = res["likely"]
            if likely == "win":
                standings[home]["pts"] += 3; standings[home]["won"] += 1
                standings[away]["lost"] += 1
            elif likely == "draw":
                standings[home]["pts"] += 1; standings[home]["drawn"] += 1
                standings[away]["pts"] += 1; standings[away]["drawn"] += 1
            else:
                standings[away]["pts"] += 3; standings[away]["won"] += 1
                standings[home]["lost"] += 1

        ranked = sorted(grp_teams, key=lambda t: standings[t]["pts"], reverse=True)

        # Standings table
        st.markdown("#### Predicted Standings")
        for rank, team in enumerate(ranked, 1):
            s = standings[team]
            qualify_badge = ' <span style="background:rgba(34,197,94,0.2);color:#22c55e;font-size:0.65rem;padding:2px 8px;border-radius:999px;font-family:\'JetBrains Mono\',monospace;">ADVANCES</span>' if rank <= 2 else ""
            st.markdown(f"""
            <div class="acc-row {'correct' if rank <= 2 else 'wrong'}" style="margin-bottom:0.5rem;">
              <span style="font-family:'JetBrains Mono',monospace;font-size:0.9rem;width:24px;color:var(--muted)">{rank}</span>
              <span class="acc-match">{team}{qualify_badge}</span>
              <span class="acc-pred">W {s['won']}  D {s['drawn']}  L {s['lost']}</span>
              <span style="font-family:'JetBrains Mono',monospace;font-weight:700;color:var(--gold)">{s['pts']} pts</span>
            </div>
            """, unsafe_allow_html=True)

        # Match predictions
        st.markdown("#### Match Predictions")
        for home, away, res in match_preds:
            p_win  = res["win"] * 100
            p_draw = res["draw"] * 100
            p_loss = res["loss"] * 100
            winner_label = home if res["likely"] == "win" else (away if res["likely"] == "loss" else "Draw")
            col_a, col_b, col_c = st.columns([3, 2, 3])
            with col_a:
                bold = "**" if res["likely"] == "win" else ""
                st.markdown(f"{bold}{home}{bold}")
                st.caption(f"{p_win:.0f}% win")
            with col_b:
                st.markdown(f"<div style='text-align:center;color:var(--muted);padding-top:4px'>Draw {p_draw:.0f}%</div>", unsafe_allow_html=True)
            with col_c:
                bold = "**" if res["likely"] == "loss" else ""
                st.markdown(f"{bold}{away}{bold}")
                st.caption(f"{p_loss:.0f}% win")

    # ── FIFA TAB 3 — TOURNAMENT SIMULATION ───────────────────────────────────
    with tab_sim_f:
        st.markdown('<div class="section-title">Tournament Simulation</div>', unsafe_allow_html=True)
        st.markdown("""
        <p style="color:var(--muted);font-size:0.85rem;margin-bottom:1.5rem;">
        Monte Carlo simulation of the entire FIFA World Cup 2026 — groups through final.
        Each run simulates match outcomes stochastically using model probabilities.
        </p>
        """, unsafe_allow_html=True)

        n_sims = st.slider("Number of Simulations", 30, 100, 50, step=10, key="fifa_n_sims")

        if st.button("🌍 Run Tournament Simulation", use_container_width=True, type="primary", key="fifa_sim_btn"):
            with st.spinner(f"Simulating {n_sims:,} World Cups…"):
                sim_result = simulate_tournament(n=n_sims)

            champ_sorted = dict(sorted(sim_result["champion"].items(), key=lambda x: -x[1]))
            final_sorted = dict(sorted(sim_result["finalist"].items(), key=lambda x: -x[1]))

            st.markdown("### 🏆 Champion Probability")
            for team, pct in champ_sorted.items():
                if pct > 0:
                    st.markdown(f"""
                    <div class="sim-row">
                      <div class="sim-team">{team}</div>
                      <div class="sim-bar-wrap">
                        <div class="sim-bar" style="width:{min(pct*2,100):.0f}%"></div>
                      </div>
                      <div class="sim-pct">{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("### 🥈 Finalist Probability")
            for team, pct in final_sorted.items():
                if pct > 0:
                    st.markdown(f"""
                    <div class="sim-row">
                      <div class="sim-team">{team}</div>
                      <div class="sim-bar-wrap">
                        <div class="sim-bar" style="width:{min(pct*1.5,100):.0f}%;opacity:0.7"></div>
                      </div>
                      <div class="sim-pct">{pct}%</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.caption(f"Based on {sim_result['n']:,} simulated tournaments.")

    # ── FIFA TAB 4 — ALL FIXTURES ─────────────────────────────────────────────
    with tab_fixtures_f:
        st.markdown('<div class="section-title">All Group Stage Fixtures</div>', unsafe_allow_html=True)

        # Group filter
        grp_filter = st.selectbox(
            "Filter by group",
            ["All Groups"] + [f"Group {g}" for g in sorted(FIFA_GROUPS.keys())],
            key="fifa_fix_grp"
        )

        # Build group lookup
        team_to_group = {}
        for grp, teams in FIFA_GROUPS.items():
            for t in teams:
                team_to_group[t] = grp

        shown = 0
        for home, away, date in FIFA_GROUP_FIXTURES:
            grp = team_to_group.get(home, "?")
            if grp_filter != "All Groups" and f"Group {grp}" != grp_filter:
                continue
            res = predict_fifa(home, away)
            likely = res["likely"]
            winner_label = home if likely == "win" else (away if likely == "loss" else "Draw")
            p_win  = res["win"]  * 100
            p_draw = res["draw"] * 100
            p_loss = res["loss"] * 100
            col_a, col_b, col_c, col_d = st.columns([2, 3, 2, 2])
            with col_a:
                st.caption(f"Group {grp} · {date}")
            with col_b:
                h_bold = "**" if likely == "win" else ""
                a_bold = "**" if likely == "loss" else ""
                st.markdown(f"{h_bold}{home}{h_bold}  ·  {a_bold}{away}{a_bold}")
            with col_c:
                outcome_color = "#22c55e" if likely != "draw" else "#f59e0b"
                st.markdown(
                    f"<span style='color:{outcome_color};font-family:JetBrains Mono,monospace;font-size:0.8rem'>"
                    f"{'🏆 ' + winner_label if likely != 'draw' else '🤝 Draw'}</span>",
                    unsafe_allow_html=True
                )
            with col_d:
                st.caption(f"W {p_win:.0f}% D {p_draw:.0f}% L {p_loss:.0f}%")
            shown += 1

        st.caption(f"Showing {shown} fixtures.")

    # ── FIFA TAB 5 — ACCURACY ─────────────────────────────────────────────────
    with tab_acc_f:
        st.markdown('<div class="section-title">FIFA Model Accuracy</div>', unsafe_allow_html=True)
        render_accuracy_tab("fifa")