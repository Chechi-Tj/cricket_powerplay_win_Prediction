# featured_engineering.py
import os
import yaml
import pandas as pd
from config import RAW_PATH, OUTPUT_PATH, OVER_OUTPUT_PATH

def extract_match_features(yaml_file):
    """Extract per-innings and over-by-over features from a YAML match file."""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    match_id = os.path.basename(yaml_file).replace(".yaml", "")

    info = data.get('info', {})
    season = info.get('season')
    if not season:
        raw_date = info.get('dates', [None])[0]
        if raw_date:
            season = int(str(raw_date).split("-")[0])
    winner = info.get('outcome', {}).get('winner')

    innings_list = data.get('innings', [])
    innings_data = []
    over_data = []

    for inning in innings_list:
        for inn_name, details in inning.items():
            batting_team_name = details.get('team')  # actual team name
            deliveries = details.get('deliveries', [])

            total_runs = total_wickets = 0
            pp_runs = pp_wickets = 0
            middle_runs = middle_wickets = 0
            death_runs = death_wickets = 0

            for ball in deliveries:
                for ball_no, event in ball.items():
                    ball_key = str(ball_no)
                    if '.' in ball_key:
                        over = int(ball_key.split('.')[0]) + 1
                    else:
                        over = int(ball_key)

                    runs = event.get('runs', {}).get('total', 0)
                    wicket = 1 if 'wicket' in event else 0

                    total_runs += runs
                    total_wickets += wicket

                    # Determine phase
                    if 1 <= over <= 6:
                        phase = 'Powerplay'
                        pp_runs += runs
                        pp_wickets += wicket
                    elif 7 <= over <= 15:
                        phase = 'Middle'
                        middle_runs += runs
                        middle_wickets += wicket
                    else:
                        phase = 'Death'
                        death_runs += runs
                        death_wickets += wicket

                    # Build over-by-over record
                    over_data.append({
                        "match_id": match_id,
                        "season": season,
                        "inning": inn_name,
                        "batting_team": batting_team_name,
                        "over": over,
                        "phase": phase,
                        "runs": runs,
                        "wickets": wicket,
                        "winner": winner
                    })

            # Build per-innings summary
            innings_data.append({
                "match_id": match_id,
                "season": season,
                "inning": inn_name,
                "batting_team": batting_team_name,
                "winner": winner,
                "total_runs": total_runs,
                "total_wickets": total_wickets,
                "powerplay_runs": pp_runs,
                "powerplay_wickets": pp_wickets,
                "middle_overs_runs": middle_runs,
                "middle_overs_wickets": middle_wickets,
                "death_overs_runs": death_runs,
                "death_overs_wickets": death_wickets
            })

    print(f"✅ {yaml_file}: {len(innings_data)} innings, {len(over_data)} over records extracted")
    return innings_data, over_data


def generate_dataset():
    all_matches = []
    all_overs = []

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"RAW PATH not found: {RAW_PATH}")

    files = [f for f in os.listdir(RAW_PATH) if f.endswith(".yaml")]
    if not files:
        print(f"⚠️ No YAML files found in {RAW_PATH}")
        return

    for file in files:
        file_path = os.path.join(RAW_PATH, file)
        innings_data, over_data = extract_match_features(file_path)
        all_matches.extend(innings_data)
        all_overs.extend(over_data)

    # Save per-innings dataset
    df_matches = pd.DataFrame(all_matches)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_matches.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ IPL Master Dataset created: {OUTPUT_PATH}")

    # Save over-by-over dataset
    df_overs = pd.DataFrame(all_overs)
    os.makedirs(os.path.dirname(OVER_OUTPUT_PATH), exist_ok=True)
    df_overs.to_csv(OVER_OUTPUT_PATH, index=False)
    print(f"✅ IPL Over-by-Over Dataset created: {OVER_OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
