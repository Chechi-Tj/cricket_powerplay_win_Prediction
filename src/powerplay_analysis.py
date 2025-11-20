import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import datetime, date
# --------------------------
# Step 1: Load YAML dataset
# --------------------------
def load_cricsheet(path="../data/raw"):
    """
    Reads all YAML match files in the specified folder.
    Extracts ball-by-ball info: match_id, batting team, over, runs, wicket.
    Returns a pandas DataFrame.
    """
    records = []

    # Loop over each file in the raw data folder
    for file in os.listdir(path):
        if file.endswith(".yaml"):
            with open(os.path.join(path, file), 'r') as f:
                data = yaml.safe_load(f)  # Load YAML content

            match_id = file.replace(".yaml", "")  # Unique match identifier

            # <<< ADDED >>> extract match year
            try:
                year = int(str(data["info"]["dates"][0]).split("-")[0])
            except:
                year = None

            # Loop over innings in the match
            innings_list = data['innings']
            for inn in innings_list:
                inn_name = list(inn.keys())[0]  # '1st innings', '2nd innings', etc.
                deliveries = inn[inn_name]['deliveries']
                batting_team = inn[inn_name]['team']
                raw_date = data['info']['dates'][0]
                if isinstance(raw_date, str):
                    year = int(raw_date.split("-")[0])
                elif isinstance(raw_date, (datetime, date)):
                    year = raw_date.year
                else:
                    year = None
                # Loop over each delivery in the innings
                for ball in deliveries:
                    over_ball_key, info = next(iter(ball.items()))

                     # convert key to string to prevent float issues
                    key_str = str(over_ball_key)

                    if "." in key_str:
                        over = int(key_str.split('.')[0])
                        ball_no = int(key_str.split('.')[1])
                    else:
                        over = int(key_str)
                        ball_no = 0

                    runs = info['runs']['total']
                    wicket = 1 if 'wicket' in info else 0
                    team = inn[inn_name]['team']

                   # Append record
                    records.append({
                    'match_id': match_id,
                    'team': batting_team,
                    'over': over,
                    'runs': runs,
                    'wicket': wicket,
                    'year': year
                 })

    # Return DataFrame containing all deliveries
    return pd.DataFrame(records)

# Load all YAML files
df = load_cricsheet()
print(f"Loaded {len(df)} deliveries.")
print(df.head())

# --------------------------
# Step 2: Extract match results (winners)
# --------------------------
results = []

for file in os.listdir("../data/raw"):
    if file.endswith(".yaml"):
        with open(os.path.join("../data/raw", file), 'r') as f:
            data = yaml.safe_load(f)

        match_id = file.replace(".yaml", "")
        # Extract match winner from YAML info
        winner = data['info'].get('outcome', {}).get('winner', None)

        results.append({
            'match_id': match_id,
            'winner': winner
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# --------------------------
# Step 3: Compute Powerplay runs
# --------------------------
# Powerplay = overs 1–6 (0 to 5 in zero-indexed YAML)
pp_stats = (
    df[df['over'] < 6]
        .groupby(['match_id', 'team', 'year'])
        .agg({'runs': 'sum','wicket': 'sum'})
        .reset_index()
)

pp_stats.rename(columns={
    'runs': 'powerplay_runs',
    'wicket': 'powerplay_wkts'
}, inplace=True)


# Merge Powerplay runs with match results
final = pp_stats.merge(results_df, on='match_id', how='left')
final['won'] = (final['team'] == final['winner']).astype(int)

# --------------------------
# Step 4: Bucket Powerplay scores
# --------------------------
# Define score ranges (buckets) for analysis
pp_stats['pp_bucket'] = pd.cut(
    pp_stats['powerplay_runs'],
    bins = [0, 20, 30, 40, 50, 60,70, 200],  # last bin covers all higher scores
    labels = ['0-20','21-30','31-40','41-50','51-60','60-70','70+']
)
final = final.merge(
    pp_stats[['match_id','team','pp_bucket']],
    on=['match_id','team'],
    how='left'
)

# ============================
# NEW BLOCK A — Correlation Strength Yearly
# ============================
corr_results = (
    final.groupby('year')[['powerplay_runs', 'powerplay_wkts', 'won']]
         .corr()['won']
         .reset_index()
)

# Save the correlation results to CSV
corr_results.to_csv("../output/tables/correlation_runs_wkts_yearly.csv", index=False)
print("\n=== Correlation of Runs & Wickets with Wins, Yearly ===")
print(corr_results)

# Filter only the correlations of interest (runs & wickets with won)
corr_plot = corr_results[corr_results['level_1'].isin(['powerplay_runs', 'powerplay_wkts'])]

plt.figure(figsize=(10,5))
sns.lineplot(
    data=corr_plot,
    x='year',
    y='won',
    hue='level_1',
    marker='o'
)
plt.title("Yearly Correlation of Powerplay Runs and Wickets with Wins")
plt.xlabel("Year")
plt.ylabel("Correlation with Win")
plt.legend(title="Factor", labels=["Powerplay Runs", "Powerplay Wickets"])
plt.ylim(-1,1)
plt.tight_layout()
plt.savefig("../output/figures/corr_runs_wkts_yearly.png")
plt.show()

# NEW BLOCK B — Logistic Regression per Year
# ===========================
from sklearn.linear_model import LogisticRegression
import numpy as np

logit_results = []

for yr, grp in final.groupby('year'):
    X = grp[['powerplay_runs', 'powerplay_wkts']].fillna(0)
    y = grp['won']

    model = LogisticRegression()
    model.fit(X, y)

    logit_results.append({
        'year': yr,
        'coef_runs': model.coef_[0][0],
        'coef_wkts': model.coef_[0][1]
    })

logit_df = pd.DataFrame(logit_results)
logit_df.to_csv("../output/tables/logit_coefficients_yearly.csv", index=False)

print("\n=== Logistic Regression Coefficients ===")
print(logit_df)



plt.figure(figsize=(10,5))
sns.lineplot(
    data=logit_df,
    x='year',
    y='coef_runs',
    marker='o',
    label='Powerplay Runs'
)
sns.lineplot(
    data=logit_df,
    x='year',
    y='coef_wkts',
    marker='o',
    label='Powerplay Wickets'
)
plt.title("Yearly Logistic Regression Coefficients: Predicting Win from Powerplay Runs & Wickets")
plt.xlabel("Year")
plt.ylabel("Coefficient")
plt.legend()
plt.tight_layout()
plt.savefig("../output/figures/logit_coefficients_yearly.png")
plt.show()
# ============================
# NEW BLOCK C — AUC Comparison Yearly
# ============================
from sklearn.metrics import roc_auc_score

auc_list = []

for yr, grp in final.groupby('year'):

    # Runs as predictor
    auc_runs = roc_auc_score(grp['won'], grp['powerplay_runs'])

    # Wickets as predictor
    auc_wkts = roc_auc_score(grp['won'], grp['powerplay_wkts'] * -1)
    # invert wickets because more wickets = lower win probability

    auc_list.append({
        'year': yr,
        'AUC_runs': auc_runs,
        'AUC_wkts': auc_wkts
    })

auc_df = pd.DataFrame(auc_list)
# Save to CSV
auc_df.to_csv("../output/tables/auc_comparison_yearly.csv", index=False)

print("\n=== AUC Comparison (Runs vs Wickets) ===")
print(auc_df)

plt.figure(figsize=(10,5))
sns.lineplot(data=auc_df, x='year', y='AUC_runs', marker='o', label='Powerplay Runs')
sns.lineplot(data=auc_df, x='year', y='AUC_wkts', marker='o', label='Powerplay Wickets')
plt.title("Yearly AUC: Predicting Win from Powerplay Runs vs Wickets")
plt.xlabel("Year")
plt.ylabel("AUC")
plt.ylim(0.5, 1.0)  # AUC ranges from 0.5 to 1 for meaningful predictive power
plt.legend()
plt.tight_layout()
plt.savefig("../output/figures/auc_comparison_yearly.png")
plt.show()

# >>> NEW: Overall Win % by Powerplay Runs
win_rate = (
    final.groupby('pp_bucket')['won']
         .mean()
         .reset_index()
)
win_rate['win_rate'] = win_rate['won'] * 100
win_rate.to_csv("../output/tables/powerplay_win_rate.csv", index=False)

plt.figure(figsize=(8,4))
sns.barplot(x='pp_bucket', y='win_rate', hue='pp_bucket', data=win_rate, palette='Blues_d', legend=False)
plt.xlabel("Powerplay Runs")
plt.ylabel("Win Percentage")
plt.title("Overall Win % by Powerplay Runs")
plt.ylim(0,100)
plt.tight_layout()
plt.savefig("../output/figures/powerplay_win_rate.png")
plt.show()

# >>> NEW: Year-wise Win % by Powerplay Runs
yearly = final.groupby(['year','pp_bucket'])['won'].mean().reset_index()
yearly['win_rate'] = yearly['won']*100
yearly.to_csv("../output/tables/powerplay_win_rate_yearly.csv", index=False)

pivot = yearly.pivot(index='year', columns='pp_bucket', values='win_rate')
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label':'Win %'})
plt.title("Year-wise Win % by Powerplay Runs")
plt.tight_layout()
plt.savefig("../output/figures/powerplay_yearly_heatmap.png")
plt.show()
# --------------------------
# Win % by wickets lost in Powerplay
# --------------------------
# Bucket wickets: 0,1,2,3+
final['wicket_bucket'] = pd.cut(
    final['powerplay_wkts'],
    bins=[-1,0,1,2,100],
    labels=['0 wkts', '1 wkt', '2 wkts', '3+ wkts']
)

wk_rate = final.groupby('wicket_bucket')['won'].mean().reset_index()
wk_rate['win_rate'] = wk_rate['won'] * 100
wk_rate.to_csv("../output/tables/powerplay_win_rate_by_wickets.csv", index=False)

# Plot
plt.figure(figsize=(7,4))
sns.barplot(x='wicket_bucket', y='win_rate', data=wk_rate)
plt.ylim(0,100)
plt.title("Win % vs Wickets Lost in Powerplay")
plt.xlabel("Wickets Lost in Powerplay")
plt.ylabel("Win Percentage")
plt.tight_layout()
plt.savefig("../output/figures/powerplay_wicket_impact.png")
plt.show()

# >>> NEW: Overall Win % by Powerplay Runs AND Wickets
final['wicket_bucket'] = pd.cut(
    final['powerplay_wkts'],
    bins=[-1,0,1,2,100],
    labels=['0 wkts','1 wkt','2 wkts','3+ wkts']
)

pp_wk_win = (
    final.groupby(['pp_bucket','wicket_bucket'])['won']
         .mean()
         .reset_index()
)
pp_wk_win['win_rate'] = pp_wk_win['won']*100
pp_wk_win.to_csv("../output/tables/powerplay_win_by_runs_wickets.csv", index=False)

pivot = pp_wk_win.pivot(index='pp_bucket', columns='wicket_bucket', values='win_rate')
plt.figure(figsize=(8,5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label':'Win %'})
plt.title("Win % by Powerplay Runs and Wickets Lost")
plt.xlabel("Wickets Lost in Powerplay")
plt.ylabel("Powerplay Runs Bucket")
plt.tight_layout()
plt.savefig("../output/figures/powerplay_runs_wickets_heatmap.png")
plt.show()
# --------------------------
## >>> NEW: Year-wise Win % by Powerplay Runs AND Wickets
pp_wk_yearly = (
    final.groupby(['year','pp_bucket','wicket_bucket'])['won']
         .mean()
         .reset_index()
)
pp_wk_yearly['win_rate'] = pp_wk_yearly['won']*100
pp_wk_yearly.to_csv("../output/tables/powerplay_win_by_runs_wickets_yearly.csv", index=False)

for yr in sorted(final['year'].dropna().unique()):
    pivot = pp_wk_yearly[pp_wk_yearly['year']==yr].pivot(index='pp_bucket', columns='wicket_bucket', values='win_rate')
    plt.figure(figsize=(8,5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label':'Win %'})
    plt.title(f"Year {yr}: Win % by Powerplay Runs and Wickets")
    plt.xlabel("Wickets Lost in Powerplay")
    plt.ylabel("Powerplay Runs Bucket")
    plt.tight_layout()
    plt.savefig(f"../output/figures/powerplay_runs_wickets_heatmap_{yr}.png")
    plt.close()

print("All analyses complete! CSVs and figures saved.")