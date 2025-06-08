import requests
import pandas as pd
import os

def fetch_squad_table(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    tables = pd.read_html(response.text, attrs={"class": "items"})
    if tables:
        squad_df = tables[0]
        return squad_df
    else:
        raise ValueError("No squad table found on the page.")

def clean_and_split_player_position(df):
    # Find the column that contains player and position info
    player_col = [col for col in df.columns if 'Player' in col][0]
    # Split after the last space (handles multiple spaces in name)
    df[['Name', 'Position']] = df[player_col].str.extract(r'^(.*)\s+([^\s]+)$')
    df = df.drop(columns=[player_col])
    return df

def process_teams(team_urls):
    all_squads = []
    for url in team_urls:
        squad_df = fetch_squad_table(url)
        squad_df = squad_df.dropna(subset=['Foot'])
        squad_df = clean_and_split_player_position(squad_df)
        squad_df['Team'] = url.split('/')[3]  # Extract team name from URL
        all_squads.append(squad_df)
    return pd.concat(all_squads, ignore_index=True)

if __name__ == "__main__":
    team_urls = [
        "https://www.transfermarkt.co.uk/tottenham-hotspur/kader/verein/148/saison_id/2024/plus/1",
        "https://www.transfermarkt.co.uk/brentford-fc/kader/verein/1148/saison_id/2024/plus/1"
    ]
    combined_df = process_teams(team_urls)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'transfermarkt_data.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Combined squad data saved to {output_path}")
