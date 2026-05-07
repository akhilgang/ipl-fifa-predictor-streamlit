# IPL 2025 AI Predictor · Streamlit

## Folder Structure

```
ipl_app/
├── app.py                        ← main Streamlit app
├── style.css                     ← custom dark theme CSS
├── requirements.txt
├── models/                       ← PUT YOUR .pkl FILES HERE
│   ├── ipl_model.pkl
│   ├── ipl_le_t1.pkl
│   ├── ipl_le_t2.pkl
│   └── ipl_le_venue.pkl
└── data/
    ├── ipl_teams.json            ← team list
    ├── ipl_fixtures.json         ← all 74 matches with results
    ├── ipl_points_table.json     ← final league stage table
    └── accuracy_log.json         ← auto-created when you log results
```

## Step 1 — Copy your pkl files

```bash
cp ipl_model.pkl     ipl_app/models/
cp ipl_le_t1.pkl     ipl_app/models/
cp ipl_le_t2.pkl     ipl_app/models/
cp ipl_le_venue.pkl  ipl_app/models/
```

## Step 2 — Install dependencies

```bash
cd ipl_app
pip install -r requirements.txt
```

## Step 3 — Run locally

```bash
streamlit run app.py
```

Open http://localhost:8501

## Step 4 — Deploy to Streamlit Cloud (free)

1. Push the `ipl_app/` folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click **New app** → connect your GitHub repo
4. Set **Main file path** = `app.py`
5. Add secret in **Advanced settings**:
   - Key: `ADMIN_PASSWORD`  Value: `your_secret_password`
6. Click **Deploy**

Done — your app is live at `https://your-app.streamlit.app`

## Step 5 — Use the Accuracy feature

### Auto-method (easiest):
After each prediction in the app, a "Who actually won?" radio appears below the result.
Select the actual winner and click **Save Result** — it logs to `accuracy_log.json`.

### Bulk seed from fixtures (run once):
The fixtures JSON already has `actual_winner` for all 74 matches.
Run this once to pre-populate the accuracy log:

```bash
python seed_accuracy.py
```

## Environment Variables

| Variable         | Default        | Purpose                        |
|------------------|----------------|--------------------------------|
| ADMIN_PASSWORD   | `ipl2025admin` | Password for Admin tab         |

Set via `.streamlit/secrets.toml` locally:
```toml
ADMIN_PASSWORD = "your_strong_password"
```

Or via Streamlit Cloud → App settings → Secrets.

## Notes

- `accuracy_log.json` is written locally. On Streamlit Cloud, it resets on redeploy.
  For persistent accuracy tracking, swap `save_accuracy()` to write to a hosted DB
  (Supabase free tier works great with `st.secrets`).
- The simulation tab uses the `ipl_fixtures.json` — mark matches as `"played": true`
  via the Admin tab to exclude them from remaining fixtures.
- IPL 2025 is fully completed (RCB won). The fixtures file has all results pre-filled.
