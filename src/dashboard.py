# src/dashboard_ml_advanced.py
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "ipl_master_dataset.csv"
RAW_DIR = ROOT / "data" / "raw"
OVER_BY_OVER_PATH = ROOT / "data" / "processed" / "over_by_over.csv"
MODEL_PATH = ROOT / "models" / "pp_win_model.joblib"
os.makedirs(ROOT / "models", exist_ok=True)

# --------- HELPERS ----------
@st.cache_data
def load_master():
    if not PROCESSED.exists():
        st.error(f"Processed dataset not found at {PROCESSED}")
        st.stop()
    df = pd.read_csv(PROCESSED)
    if "won" not in df.columns:
        df["won"] = (df["batting_team"] == df["winner"]).astype(int)
    return df

def build_over_by_over(raw_dir=RAW_DIR, out_path=OVER_BY_OVER_PATH):
    """Builds over_by_over CSV: one row per match, inning, over -> runs_this_over, includes season"""
    records = []
    for f in os.listdir(raw_dir):
        if not f.endswith(".yaml"):
            continue
        file_path = raw_dir / f
        match_id = Path(f).stem
        with open(file_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        info = data.get("info", {})
        season = info.get("season")
        if not season:
            raw_date = info.get("dates", [None])[0]
            if raw_date:
                season = int(str(raw_date).split("-")[0])
        for inning in data.get("innings", []):
            for inn_name, details in inning.items():
                batting = details.get("team")
                deliveries = details.get("deliveries", [])
                over_runs = {}
                for ball in deliveries:
                    k, event = next(iter(ball.items()))
                    k_str = str(k)
                    if "." in k_str:
                        over_idx = int(k_str.split(".")[0]) + 1
                    else:
                        over_idx = int(k_str)
                    runs = event.get("runs", {}).get("total", 0)
                    over_runs.setdefault(over_idx, 0)
                    over_runs[over_idx] += runs
                for over, runs in sorted(over_runs.items()):
                    records.append({
                        "match_id": match_id,
                        "season": season,
                        "inning_label": inn_name,
                        "batting_team": batting,
                        "over": over,
                        "runs": runs
                    })
    df_over = pd.DataFrame(records)
    os.makedirs(out_path.parent, exist_ok=True)
    df_over.to_csv(out_path, index=False)
    return df_over

@st.cache_data
def load_over_by_over():
    if OVER_BY_OVER_PATH.exists():
        return pd.read_csv(OVER_BY_OVER_PATH)
    else:
        return build_over_by_over()

# --------- ML ----------
@st.cache_data
def train_model(df):
    feats = [
        "powerplay_runs", "powerplay_wickets",
        "middle_overs_runs", "middle_overs_wickets",
        "death_overs_runs", "death_overs_wickets"
    ]
    X = df[feats].fillna(0)
    y = df["won"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    yhat = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, yhat)
    joblib.dump((model, feats), MODEL_PATH)
    return model, feats, auc

def load_model():
    if MODEL_PATH.exists():
        model, feats = joblib.load(MODEL_PATH)
        return model, feats
    return None, None

# --------- Advanced Metrics ----------
def compute_powerplay_efficiency(df):
    league_avg = df["powerplay_runs"].mean()
    df["pp_efficiency"] = df["powerplay_runs"] / (league_avg + 1e-6)
    return df, league_avg

def detect_momentum(over_df, threshold=8):
    if over_df.empty:
        return []
    s = over_df.set_index("over")["runs"].sort_index()
    r = s.rolling(2).sum().dropna()
    deltas = r.diff().dropna()
    shifts = [(int(idx), float(val)) for idx, val in deltas.items() if val >= threshold]
    return shifts

# --------- Streamlit App ----------
st.set_page_config(page_title="IPL ML + OverPlot + Advanced", layout="wide")
st.title("🏏 IPL — Win Predictor, Over-by-Over Viewer & Advanced Metrics")

df = load_master()

st.sidebar.header("Model")
if st.sidebar.button("Train model (LogisticRegression)"):
    model, feats, auc = train_model(df)
    st.sidebar.success(f"Trained model saved. AUC = {auc:.3f}")
else:
    model, feats = load_model()
    if model is None:
        st.sidebar.info("No pre-trained model. Click 'Train model' to create one.")
    else:
        st.sidebar.success("Loaded trained model.")

# ---------- Win prediction UI ----------
st.sidebar.header("Predict Win Probability")
pp_runs = st.sidebar.slider("Powerplay runs (1-6)", 0, 120, 40)
pp_wkts = st.sidebar.slider("Powerplay wickets", 0, 6, 1)
mid_runs = st.sidebar.slider("Middle overs runs (7-15)", 0, 200, 60)
mid_wkts = st.sidebar.slider("Middle overs wickets", 0, 6, 1)
death_runs = st.sidebar.slider("Death overs runs (16-20)", 0, 200, 40)
death_wkts = st.sidebar.slider("Death overs wickets", 0, 6, 1)

if model is not None:
    sample = pd.DataFrame([[pp_runs, pp_wkts, mid_runs, mid_wkts, death_runs, death_wkts]], columns=feats)
    prob = model.predict_proba(sample)[0, 1]
    st.sidebar.metric("Predicted Win Probability", f"{prob*100:.1f}%")
    coefs = pd.Series(model.coef_[0], index=feats)
    st.sidebar.subheader("Model coefficients")
    st.sidebar.table(coefs.sort_values(ascending=False).round(3))
else:
    st.sidebar.write("Train the model to get predictions.")

# ---------- Over-by-over viewer ----------
st.header("Over-by-Over Viewer")
over_df = load_over_by_over()

# --- Updated: season + batting_team selection ---
seasons = sorted(over_df["season"].dropna().unique())
sel_season = st.selectbox("Select Season", options=seasons)

teams = sorted(over_df[over_df["season"] == sel_season]["batting_team"].unique())
sel_team = st.selectbox("Select Team", options=teams)

match_rows = over_df[(over_df["season"] == sel_season) & (over_df["batting_team"] == sel_team)]
innings_options = sorted(match_rows["inning_label"].unique())
sel_inning = st.selectbox("Select Inning", options=innings_options)

ov = match_rows[match_rows["inning_label"] == sel_inning].sort_values("over")
fig = go.Figure()
fig.add_trace(go.Bar(x=ov["over"], y=ov["runs"], name="Runs / over"))
fig.update_layout(title=f"Season {sel_season} — {sel_team} — {sel_inning}", xaxis_title="Over", yaxis_title="Runs")
st.plotly_chart(fig, use_container_width=True)

shifts = detect_momentum(ov, threshold=8)
if shifts:
    st.warning(f"Momentum shifts detected (over, delta runs): {shifts}")
else:
    st.info("No significant momentum shifts detected (threshold=8 runs per 2-over window).")

# ---------- Advanced analytics ----------
st.header("Advanced Analytics (Per-innings metrics)")
league_pp_avg = df["powerplay_runs"].mean()
league_death_avg = df["death_overs_runs"].mean()
df["pp_efficiency"] = df["powerplay_runs"] / (league_pp_avg + 1e-6)
df["death_collapse_index"] = (league_death_avg - df["death_overs_runs"]) / (league_death_avg + 1e-6)

st.subheader("Powerplay Efficiency")
st.dataframe(df[["match_id","inning","batting_team","powerplay_runs","pp_efficiency"]].sort_values("pp_efficiency", ascending=False).head(10))

st.subheader("Death Collapse Index")
st.dataframe(df[["match_id","inning","batting_team","death_overs_runs","death_collapse_index"]].sort_values("death_collapse_index", ascending=False).head(10))

st.markdown("""
**Definitions**
- **Powerplay Efficiency** = (this innings PP runs) / (league average PP runs). Values >1 mean above-average performance.
- **Death Collapse Index** = (league avg death runs - this innings death runs) / league avg death runs. Positive values indicate a collapse.
- **Momentum shift**: jump in 2-over rolling runs > threshold (default 8).
""")

st.success("All tools ready — explore matches & ML predictions!")
