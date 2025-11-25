import os
import yaml
import pandas as pd
from config import RAW_PATH, OUTPUT_PATH

def extract_match_features(yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    match_id = os.path.basename(yaml_file).replace(".yaml", "")

    info = data.get('info', {})
    season = info.get('season')
    teams = info.get('teams', [])
    # <<< ADD THIS HERE >>>
    if not season:
        raw_date = info.get('dates', [None])[0]
        if raw_date:
            season = int(str(raw_date).split("-")[0])
    winner = info.get('outcome', {}).get('winner')

    innings_list = data.get('innings', [])
    innings_data = []

    for inning in innings_list:
        for inn_name, details in inning.items():  # keep innings name
            batting_team_name = details.get('team')  # actual team name
            deliveries = details.get('deliveries', [])

            total_runs = 0
            total_wickets = 0

            # Phase stats
            pp_runs = pp_wickets = 0
            middle_runs = middle_wickets = 0
            death_runs = death_wickets = 0

            for ball in deliveries:
                for ball_no, event in ball.items():
                    # Parse over correctly
                    ball_key = str(ball_no)
                    if '.' in ball_key:
                        over = int(ball_key.split('.')[0]) + 1  # over numbers start from 1
                    else:
                        over = int(ball_key)

                    runs = event['runs']['total']
                    wicket = 1 if 'wicket' in event else 0

                    total_runs += runs
                    total_wickets += wicket

                    # Powerplay: 1–6 overs
                    if 1 <= over <= 6:
                        pp_runs += runs
                        pp_wickets += wicket
                    # Middle overs: 7–15
                    elif 7 <= over <= 15:
                        middle_runs += runs
                        middle_wickets += wicket
                    # Death overs: 16–20
                    else:
                        death_runs += runs
                        death_wickets += wicket

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

    return innings_data


def generate_dataset():
    all_matches = []

    if not os.path.exists(RAW_PATH):
        raise FileNotFoundError(f"RAW PATH not found: {RAW_PATH}")

    for file in os.listdir(RAW_PATH):
        if file.endswith(".yaml"):
            file_path = os.path.join(RAW_PATH, file)
            innings_data = extract_match_features(file_path)
            all_matches.extend(innings_data)  # extend to keep multiple innings

    df = pd.DataFrame(all_matches)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"✅ IPL Master Dataset created: {OUTPUT_PATH}")


if __name__ == "__main__":
    generate_dataset()
