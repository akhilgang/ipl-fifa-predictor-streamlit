"""
seed_accuracy.py
Run ONCE after deploying to pre-populate accuracy_log.json
from the actual results already in ipl_fixtures.json.

Usage:  python seed_accuracy.py
"""

import json, joblib, sys
from pathlib import Path

BASE  = Path(__file__).parent
DATA  = BASE / "data"
MODEL = BASE / "models"

def _enc(le, val, fallback=0):
    try:    return int(le.transform([val])[0])
    except: return fallback

IPL_STAGE_W = {"league":1, "qualifier":2, "eliminator":2, "final":3}

def main():
    print("Loading models…")
    model    = joblib.load(MODEL / "ipl_model.pkl")
    le_t1    = joblib.load(MODEL / "ipl_le_t1.pkl")
    le_t2    = joblib.load(MODEL / "ipl_le_t2.pkl")
    le_venue = joblib.load(MODEL / "ipl_le_venue.pkl")

    with open(DATA / "ipl_fixtures.json") as f:
        fixtures = json.load(f)

    with open(DATA / "ipl_points_table.json") as f:
        pts = json.load(f)

    def wr(team):
        p = pts.get(team, {})
        return round(p["won"]/p["played"], 4) if p.get("played",0) > 0 else 0.5

    records = []
    correct_count = 0
    skipped = 0

    for fix in fixtures:
        actual = fix.get("actual_winner")
        if not actual or fix.get("abandoned"):
            skipped += 1
            continue

        t1, t2  = fix["team1"], fix["team2"]
        venue   = fix.get("venue","")
        stage   = fix.get("stage","league")
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
        if correct == "yes":
            correct_count += 1

        records.append({
            "match_id":         fix["match_id"],
            "team1":            t1,
            "team2":            t2,
            "venue":            venue,
            "stage":            stage,
            "predicted_winner": predicted,
            "p_t1":             round(p_t1*100, 1),
            "p_t2":             round((1-p_t1)*100, 1),
            "actual_winner":    actual,
            "correct":          correct,
            "timestamp":        fix.get("date","2025-01-01") + "T20:00:00",
        })

    out = DATA / "accuracy_log.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2)

    total = len(records)
    print(f"\n✅ Seeded {total} predictions ({skipped} skipped / abandoned)")
    print(f"   Correct:   {correct_count}/{total}")
    print(f"   Accuracy:  {correct_count/total*100:.1f}%")
    print(f"\n   Saved → {out}")

if __name__ == "__main__":
    main()
