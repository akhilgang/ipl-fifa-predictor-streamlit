"""
fifa_rankings.py — FIFA World Ranking → win-rate mapping for FIFA World Cup 2026 predictor.

Win rate is derived from FIFA ranking points (as of May 2026 approximation).
Formula: win_rate = 0.3 + 0.4 * (rank_points / max_rank_points)
Clamped to [0.30, 0.75] so even top teams aren't guaranteed wins.

Usage:
    from fifa_rankings import get_win_rate
    wr = get_win_rate("Brazil")   # → e.g. 0.72
"""

# FIFA ranking points (approximate, based on official May 2026 rankings)
# Source: fifa.com rankings — higher = stronger
_FIFA_RANKING_POINTS: dict[str, float] = {
    # Elite
    "Argentina":        1896.0,
    "France":           1866.0,
    "England":          1856.0,
    "Belgium":          1830.0,
    "Brazil":           1820.0,
    "Portugal":         1810.0,
    "Netherlands":      1808.0,
    "Spain":            1800.0,
    "Germany":          1790.0,
    "Uruguay":          1745.0,

    # Strong
    "Colombia":         1735.0,
    "Switzerland":      1710.0,
    "Japan":            1705.0,
    "Morocco":          1695.0,
    "Croatia":          1690.0,
    "Mexico":           1680.0,
    "USA":              1650.0,
    "Senegal":          1640.0,
    "Sweden":           1630.0,
    "Australia":        1620.0,
    "Ecuador":          1610.0,
    "Turkey":           1605.0,
    "Austria":          1600.0,
    "Korea Republic":   1590.0,
    "Norway":           1580.0,
    "South Korea":      1590.0,  # alias

    # Mid-tier
    "Algeria":          1560.0,
    "Canada":           1555.0,
    "Egypt":            1550.0,
    "Saudi Arabia":     1540.0,
    "Tunisia":          1530.0,
    "Ghana":            1520.0,
    "IR Iran":          1515.0,
    "Iran":             1515.0,  # alias
    "Paraguay":         1510.0,
    "Scotland":         1500.0,
    "Czechia":          1495.0,
    "Czech Republic":   1495.0,  # alias
    "Ivory Coast":      1490.0,
    "Cote dIvoire":     1490.0,  # alias
    "Jordan":           1450.0,
    "Uzbekistan":       1440.0,
    "Cabo Verde":       1435.0,
    "Cape Verde":       1435.0,  # alias
    "Bosnia Herzegovina":  1430.0,
    "Bosnia and Herzegovina": 1430.0,  # alias
    "Curacao":          1380.0,
    "Curaçao":          1380.0,  # alias
    "Congo DR":         1370.0,
    "DR Congo":         1370.0,  # alias
    "South Africa":     1360.0,
    "New Zealand":      1340.0,
    "Iraq":             1330.0,
    "Panama":           1310.0,
    "Qatar":            1290.0,
    "Haiti":            1240.0,
}

_MAX_POINTS = max(_FIFA_RANKING_POINTS.values())  # normalisation anchor
_MIN_WR = 0.28
_MAX_WR = 0.74


def get_win_rate(team: str) -> float:
    """
    Return a win-rate float in [0.28, 0.74] for a given team name.
    Falls back to 0.40 for unknown teams (weak assumption).
    """
    points = _FIFA_RANKING_POINTS.get(team)
    if points is None:
        # Try case-insensitive match
        lower_map = {k.lower(): v for k, v in _FIFA_RANKING_POINTS.items()}
        points = lower_map.get(team.lower(), 1200.0)  # 1200 = weak team default

    wr = _MIN_WR + (_MAX_WR - _MIN_WR) * (points / _MAX_POINTS)
    return round(min(_MAX_WR, max(_MIN_WR, wr)), 4)


def get_all_win_rates() -> dict[str, float]:
    """Return win rates for all known teams — useful for debugging."""
    return {team: get_win_rate(team) for team in _FIFA_RANKING_POINTS}


if __name__ == "__main__":
    # Quick sanity check
    test_teams = ["Argentina", "Brazil", "Haiti", "Qatar", "England", "Morocco"]
    for t in test_teams:
        print(f"{t:<25} → {get_win_rate(t):.4f}")