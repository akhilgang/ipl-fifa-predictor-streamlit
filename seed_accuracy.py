"""
seed_accuracy.py
Run ONCE after deploying to pre-populate accuracy logs from actual results.

Usage:
  python seed_accuracy.py ipl      # seeds ipl_accuracy_log.json
  python seed_accuracy.py fifa     # seeds fifa_accuracy_log.json
  python seed_accuracy.py          # seeds both (default)
"""

import json, joblib, sys
from pathlib import Path

BASE  = Path(__file__).parent
DATA  = BASE / "data"
MODEL = BASE / "models"


def _enc(le, val, fallback=0):
    try:    return int(le.transform([val])[0])
    except: return fallback


# ─────────────────────────────────────────────────────────────────────────────
#  IPL SEEDER
# ─────────────────────────────────────────────────────────────────────────────

IPL_STAGE_W = {"league": 1, "qualifier": 2, "eliminator": 2, "final": 3}


def seed_ipl():
    print("\n── IPL ──────────────────────────────────────────")
    print("Loading IPL models…")
    model    = joblib.load(MODEL / "ipl_model.pkl")
    le_t1    = joblib.load(MODEL / "ipl_le_t1.pkl")
    le_t2    = joblib.load(MODEL / "ipl_le_t2.pkl")
    le_venue = joblib.load(MODEL / "ipl_le_venue.pkl")

    fixtures_path = DATA / "ipl_fixtures.json"
    pts_path      = DATA / "ipl_points_table.json"

    if not fixtures_path.exists():
        print("❌ ipl_fixtures.json not found — skipping IPL."); return
    if not pts_path.exists():
        print("❌ ipl_points_table.json not found — skipping IPL."); return

    with open(fixtures_path) as f: fixtures = json.load(f)
    with open(pts_path)      as f: pts      = json.load(f)

    def wr(team):
        p = pts.get(team, {})
        return round(p["won"] / p["played"], 4) if p.get("played", 0) > 0 else 0.5

    records, correct_count, skipped = [], 0, 0

    for fix in fixtures:
        actual = fix.get("actual_winner")
        if not actual or fix.get("abandoned"):
            skipped += 1
            continue

        t1, t2  = fix["team1"], fix["team2"]
        venue   = fix.get("venue", "")
        stage   = fix.get("stage", "league")
        sw      = IPL_STAGE_W.get(stage, 1)
        t1_wr   = wr(t1)
        t2_wr   = wr(t2)

        feats = [[
            _enc(le_t1, t1), _enc(le_t2, t2),
            1, 1,
            _enc(le_venue, venue) if venue else 0,
            sw,
            t1_wr, t2_wr,
            0, 0,
            t1_wr - t2_wr, 0,
            0.5, 2026,
        ]]

        proba   = model.predict_proba(feats)[0]
        classes = list(model.classes_)
        p_t1    = float(proba[classes.index(1)])
        predicted = t1 if p_t1 >= 0.5 else t2
        correct   = "yes" if predicted == actual else "no"
        if correct == "yes": correct_count += 1

        records.append({
            "match_id":         fix["match_id"],
            "team1":            t1,
            "team2":            t2,
            "venue":            venue,
            "stage":            stage,
            "predicted_winner": predicted,
            "p_t1":             round(p_t1 * 100, 1),
            "p_t2":             round((1 - p_t1) * 100, 1),
            "actual_winner":    actual,
            "correct":          correct,
            "timestamp":        fix.get("date", "2025-01-01") + "T20:00:00",
        })

    out = DATA / "ipl_accuracy_log.json"
    with open(out, "w") as f: json.dump(records, f, indent=2)

    total = len(records)
    print(f"✅ Seeded {total} IPL predictions ({skipped} skipped / abandoned)")
    if total:
        print(f"   Correct:  {correct_count}/{total}")
        print(f"   Accuracy: {correct_count/total*100:.1f}%")
    print(f"   Saved  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  FIFA SEEDER
# ─────────────────────────────────────────────────────────────────────────────

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

FIFA_TOURNAMENT_WEIGHT = 10


def resolve(name):
    return TEAM_ALIASES.get(name, name)


def seed_fifa():
    print("\n── FIFA ─────────────────────────────────────────")
    print("Loading FIFA models…")

    for p in [MODEL / "fifa_model.pkl", MODEL / "fifa_le_home.pkl", MODEL / "fifa_le_away.pkl"]:
        if not p.exists():
            print(f"❌ {p.name} not found — run train_fifa.py first. Skipping FIFA.")
            return

    model   = joblib.load(MODEL / "fifa_model.pkl")
    le_home = joblib.load(MODEL / "fifa_le_home.pkl")
    le_away = joblib.load(MODEL / "fifa_le_away.pkl")
    classes = list(model.classes_)  # e.g. ["draw", "loss", "win"]

    fixtures_path = DATA / "fifa_fixtures.json"
    if not fixtures_path.exists():
        print("ℹ️  No fifa_fixtures.json found in data/.")
        print("   This file should list completed World Cup matches with actual_winner fields.")
        print("   Expected record format:")
        print('   {"match_id":"...", "home":"Brazil", "away":"France",')
        print('    "date":"2026-06-13", "stage":"group",')
        print('    "actual_winner":"Brazil",  // or "Draw" or away team name')
        print('    "home_win_rate":0.6, "away_win_rate":0.55}')
        print("   Seed this file once real results come in during the tournament.")
        return

    with open(fixtures_path) as f:
        fixtures = json.load(f)

    records, correct_count, skipped = [], 0, 0

    for fix in fixtures:
        actual = fix.get("actual_winner")
        if not actual or fix.get("abandoned"):
            skipped += 1
            continue

        home  = fix["home"]
        away  = fix["away"]
        h_enc = _enc(le_home, resolve(home))
        a_enc = _enc(le_away, resolve(away))
        h_wr  = fix.get("home_win_rate", 0.5)
        a_wr  = fix.get("away_win_rate", 0.5)

        feats = [[
            h_enc, a_enc,
            FIFA_TOURNAMENT_WEIGHT,
            1,              # neutral venue (World Cup)
            h_wr, a_wr,
            0.0, 0.0,       # avg_gd placeholders
            h_wr - a_wr,    # wr_diff
            0.0,            # gd_diff
            0.33,           # h2h default
        ]]

        proba    = model.predict_proba(feats)[0]
        prob_map = {c: float(proba[i]) for i, c in enumerate(classes)}
        p_win    = prob_map.get("win",  0.33)
        p_draw   = prob_map.get("draw", 0.34)
        p_loss   = prob_map.get("loss", 0.33)

        # Normalise
        s = p_win + p_draw + p_loss
        p_win, p_draw, p_loss = p_win/s, p_draw/s, p_loss/s

        likely = (
            "win"  if p_win  >= p_draw and p_win  >= p_loss else
            "draw" if p_draw >= p_loss else
            "loss"
        )
        predicted_winner = home if likely == "win" else (away if likely == "loss" else "Draw")

        correct = "yes" if predicted_winner == actual else "no"
        if correct == "yes": correct_count += 1

        records.append({
            "match_id":         fix.get("match_id", f"{home}_vs_{away}"),
            "team1":            home,
            "team2":            away,
            "stage":            fix.get("stage", "group"),
            "predicted_winner": predicted_winner,
            "p_t1":             round(p_win  * 100, 1),
            "p_draw":           round(p_draw * 100, 1),
            "p_t2":             round(p_loss * 100, 1),
            "actual_winner":    actual,
            "correct":          correct,
            "timestamp":        fix.get("date", "2026-06-11") + "T20:00:00",
        })

    out = DATA / "fifa_accuracy_log.json"
    with open(out, "w") as f: json.dump(records, f, indent=2)

    total = len(records)
    print(f"✅ Seeded {total} FIFA predictions ({skipped} skipped / abandoned)")
    if total:
        print(f"   Correct:  {correct_count}/{total}")
        print(f"   Accuracy: {correct_count/total*100:.1f}%")
    print(f"   Saved  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    arg = sys.argv[1].lower() if len(sys.argv) > 1 else "both"

    if arg == "ipl":
        seed_ipl()
    elif arg == "fifa":
        seed_fifa()
    else:
        seed_ipl()
        seed_fifa()

    print("\nDone.\n")
