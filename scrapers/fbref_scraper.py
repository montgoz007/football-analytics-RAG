import os
import pandas as pd
from bs4 import BeautifulSoup
import cloudscraper

# Team URLs
TEAMS = {
    "Tottenham": "https://fbref.com/en/squads/361ca564/Tottenham-Hotspur-Stats",
    "Brentford": "https://fbref.com/en/squads/cd051869/Brentford-Stats"
}

# Set up output directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "fbref_data")
os.makedirs(DATA_DIR, exist_ok=True)

def scrape_team_tables(team, url):
    scraper = cloudscraper.create_scraper()
    response = scraper.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    tables = soup.find_all("table")
    team_tables = {}
    for idx, table in enumerate(tables):
        # Try to read as multi-header
        try:
            df = pd.read_html(str(table), header=[0, 1])[0]
            # Flatten MultiIndex columns
            df.columns = [
                f"{str(a).strip()}_{str(b).strip()}" if str(a).strip() != '' else str(b).strip()
                for a, b in df.columns
            ]
        except ValueError:
            # Fallback to single header
            df = pd.read_html(str(table), header=0)[0]
        # Remove any rows where 'Player' or similar is repeated (header rows in body)
        df = df[df[df.columns[0]] != df.columns[0]]
        df["Team"] = team  # Add team column
        table_id = table.get("id", f"table_{idx}")
        team_tables[table_id] = df.reset_index(drop=True)
    return team_tables

def merge_and_save_tables(teams):
    all_tables = {}
    for team, url in teams.items():
        team_tables = scrape_team_tables(team, url)
        for table_id, df in team_tables.items():
            if table_id not in all_tables:
                all_tables[table_id] = []
            all_tables[table_id].append(df)
    # Merge and save
    for table_id, dfs in all_tables.items():
        merged_df = pd.concat(dfs, ignore_index=True)
        filename = f"{table_id}.csv"
        filepath = os.path.join(DATA_DIR, filename)
        merged_df.to_csv(filepath, index=False)
        print(f"Saved merged table: {filename}")

if __name__ == "__main__":
    merge_and_save_tables(TEAMS)