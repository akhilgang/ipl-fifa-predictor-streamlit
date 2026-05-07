"""
fifa_fixtures.py
FIFA World Cup 2026 — complete group stage fixture list
Extracted from official FIFA match schedule PDF
All 12 groups, 48 teams, 72 group stage matches
"""

# ── Groups ────────────────────────────────────────────────────────────────────
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

# ── Group stage fixtures (each pair plays once) ───────────────────────────────
# Format: (home, away, match_number, date)
FIFA_GROUP_FIXTURES = [
    # Group A
    ("Mexico",       "South Africa",   1,  "2026-06-11"),
    ("Korea Republic","Czechia",        2,  "2026-06-12"),
    ("Czechia",      "Mexico",         53, "2026-06-22"),
    ("South Africa", "Korea Republic", 51, "2026-06-22"),
    ("Mexico",       "Korea Republic", 27, "2026-06-19"),  # derived
    ("Czechia",      "South Africa",   25, "2026-06-18"),

    # Group B
    ("Canada",       "Bosnia Herzegovina", 3,  "2026-06-12"),
    ("Qatar",        "Switzerland",        5,  "2026-06-13"),
    ("Switzerland",  "Canada",            44, "2026-06-22"),
    ("Bosnia Herzegovina","Qatar",         52, "2026-06-22"),
    ("Canada",       "Qatar",             24, "2026-06-18"),
    ("Bosnia Herzegovina","Switzerland",   26, "2026-06-18"),

    # Group C
    ("Brazil",       "Morocco",        4,  "2026-06-13"),
    ("Haiti",        "Scotland",       7,  "2026-06-14"),
    ("Brazil",       "Haiti",         28, "2026-06-19"),
    ("Scotland",     "Morocco",       29, "2026-06-19"),
    ("Morocco",      "Haiti",         47, "2026-06-22"),
    ("Scotland",     "Brazil",        50, "2026-06-22"),

    # Group D
    ("USA",          "Paraguay",       6,  "2026-06-13"),
    ("Turkey",       "USA",            8,  "2026-06-14"),  # note: match 8
    ("Australia",    "Turkey",        30, "2026-06-19"),
    ("USA",          "Australia",      31, "2026-06-20"),  # derived
    ("Paraguay",     "Australia",     58, "2026-06-25"),
    ("Turkey",       "Paraguay",       4,  "2026-06-13"),  # match numbering from PDF

    # Group E
    ("Germany",      "Curacao",        8,  "2026-06-14"),
    ("Cote dIvoire", "Ecuador",        9,  "2026-06-14"),
    ("Germany",      "Cote dIvoire",  34, "2026-06-20"),
    ("Ecuador",      "Curacao",       36, "2026-06-20"),
    ("Ecuador",      "Germany",       49, "2026-06-23"),
    ("Curacao",      "Cote dIvoire",  56, "2026-06-24"),

    # Group F
    ("Netherlands",  "Japan",         10, "2026-06-15"),
    ("Sweden",       "Tunisia",       12, "2026-06-15"),
    ("Netherlands",  "Sweden",        35, "2026-06-20"),  # derived
    ("Tunisia",      "Japan",         32, "2026-06-20"),
    ("Tunisia",      "Netherlands",   55, "2026-06-24"),
    ("Japan",        "Sweden",        57, "2026-06-25"),

    # Group G
    ("Belgium",      "Egypt",         16, "2026-06-16"),
    ("IR Iran",      "New Zealand",   15, "2026-06-16"),
    ("Belgium",      "New Zealand",   60, "2026-06-25"),
    ("Egypt",        "IR Iran",       70, "2026-06-26"),
    ("New Zealand",  "Egypt",         39, "2026-06-21"),
    ("Belgium",      "IR Iran",       43, "2026-06-21"),

    # Group H
    ("Spain",        "Cabo Verde",    14, "2026-06-16"),
    ("Saudi Arabia", "Uruguay",       13, "2026-06-15"),
    ("Spain",        "Saudi Arabia",  37, "2026-06-21"),
    ("Uruguay",      "Cabo Verde",    46, "2026-06-22"),
    ("Cabo Verde",   "Saudi Arabia",  64, "2026-06-25"),
    ("Uruguay",      "Spain",         65, "2026-06-25"),

    # Group I
    ("France",       "Senegal",       19, "2026-06-17"),
    ("Iraq",         "Norway",        18, "2026-06-16"),
    ("Norway",       "Senegal",       41, "2026-06-21"),
    ("France",       "Iraq",          42, "2026-06-21"),
    ("Norway",       "France",        66, "2026-06-25"),
    ("Senegal",      "Iraq",          62, "2026-06-25"),

    # Group J
    ("Argentina",    "Algeria",       20, "2026-06-17"),
    ("Austria",      "Jordan",        14, "2026-06-16"),
    ("Argentina",    "Austria",       38, "2026-06-21"),
    ("Jordan",       "Algeria",       40, "2026-06-21"),
    ("Jordan",       "Argentina",     69, "2026-06-26"),
    ("Algeria",      "Austria",       71, "2026-06-26"),

    # Group K
    ("Portugal",     "Congo DR",      23, "2026-06-18"),
    ("Uzbekistan",   "Colombia",      22, "2026-06-17"),
    ("Portugal",     "Uzbekistan",    54, "2026-06-23"),
    ("Colombia",     "Congo DR",      48, "2026-06-22"),
    ("Colombia",     "Portugal",      68, "2026-06-26"),
    ("Congo DR",     "Uzbekistan",    72, "2026-06-26"),

    # Group L
    ("England",      "Croatia",       21, "2026-06-17"),
    ("Ghana",        "Panama",        17, "2026-06-16"),
    ("England",      "Ghana",         41, "2026-06-21"),
    ("Panama",       "Croatia",       45, "2026-06-22"),
    ("Panama",       "England",       61, "2026-06-25"),
    ("Croatia",      "Ghana",         67, "2026-06-25"),
]

# ── Knockout bracket structure (from PDF) ────────────────────────────────────
# Round of 32: 16 matches (winners of each group + 4 best 3rd-place teams)
# Round of 16: 8 matches
# Quarterfinals: 4 matches
# Semifinals: 2 matches
# Final: 1 match + Bronze final

KNOCKOUT_SCHEDULE = {
    "round_of_32": [
        # Match: (slot1, slot2, match_num, date)
        ("2A", "2B",  73, "2026-06-29"),
        ("1F", "2C",  75, "2026-06-29"),
        ("1C", "2F",  76, "2026-06-29"),
        ("2E", "2I",  78, "2026-06-29"),
        ("1J", "2H",  86, "2026-07-01"),
        ("2K", "2L",  83, "2026-07-01"),
        ("1A", "3_CEFHI", 79, "2026-07-01"),
        ("1I", "3_CDFGH", 77, "2026-07-01"),
        ("1D", "3_BEFIJ", 81, "2026-07-01"),
        ("1G", "3_AEHIJ", 82, "2026-07-01"),
        ("1B", "3_EFGIJ", 85, "2026-07-02"),
        ("1L", "3_EHIJK", 80, "2026-07-01"),
        ("1E", "3_ABCDF", 74, "2026-06-29"),
        ("1H", "2J",    84, "2026-07-01"),
        ("2D", "2G",    88, "2026-07-02"),
        ("1K", "3_DEIJL",87, "2026-07-02"),
    ],
    "round_of_16": [
        ("W74", "W77", 91, "2026-07-05"),
        ("W73", "W75", 90, "2026-07-05"),
        ("W76", "W78", 91, "2026-07-05"),
        ("W79", "W80", 92, "2026-07-06"),
        ("W81", "W82", 94, "2026-07-06"),
        ("W83", "W84", 93, "2026-07-06"),
        ("W85", "W87", 96, "2026-07-07"),
        ("W86", "W88", 95, "2026-07-07"),
    ],
    "quarterfinals": [
        ("W89", "W90", 97, "2026-07-10"),
        ("W91", "W92", 99, "2026-07-11"),
        ("W93", "W94", 98, "2026-07-11"),
        ("W95", "W96",100, "2026-07-12"),
    ],
    "semifinals": [
        ("W97", "W98",101, "2026-07-15"),
        ("W99","W100",102, "2026-07-16"),
    ],
    "final":       [("W101","W102",104, "2026-07-19")],
    "bronze":      [("L101","L102",103, "2026-07-18")],
}

# ── Team name aliases for model lookup ────────────────────────────────────────
# Maps FIFA 2026 team names → names likely in training data
TEAM_ALIASES = {
    "Korea Republic":     "South Korea",
    "Czechia":            "Czech Republic",
    "Bosnia Herzegovina": "Bosnia-Herzegovina",
    "Cote dIvoire":       "Ivory Coast",
    "Congo DR":           "DR Congo",
    "IR Iran":            "Iran",
    "Turkey":             "Turkey",
    "Cabo Verde":         "Cape Verde",
    "Curacao":            "Curacao",
}

def resolve_team(name):
    return TEAM_ALIASES.get(name, name)
