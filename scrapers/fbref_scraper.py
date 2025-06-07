import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import os
from io import StringIO
import re

def sanitize_filename(name):
    """Sanitize filenames to remove invalid characters."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

def scrape_tables(url, header_rows=[0, 1]):
    """Scrape all tables from the given URL."""
    # Create a cloudscraper instance
    scraper = cloudscraper.create_scraper(
        browser={
            'browser': 'chrome',
            'platform': 'linux',
            'desktop': True
        }
    )

    # Fetch the HTML content of the page
    response = scraper.get(url)
    response.raise_for_status()  # Raise an error for failed requests

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all tables on the page
    tables = soup.find_all('table')

    # Extract tables into pandas DataFrames
    dataframes = []
    for table in tables:
        try:
            # Wrap the HTML string in StringIO to avoid FutureWarning
            df = pd.read_html(StringIO(str(table)), header=header_rows)[0]  # Read with multi-row headers
            # Flatten multi-row headers into single row
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
            dataframes.append(df)
        except ValueError:
            print("Skipping a table due to parsing issues.")

    return dataframes

def save_tables(tables, folder_path, table_titles=None):
    """Save tables with their respective titles in the specified folder."""
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Save each table with its corresponding title
    for i, df in enumerate(tables):
        if table_titles and i < len(table_titles):  # Use provided titles if available
            sanitized_title = sanitize_filename(table_titles[i])
        else:
            sanitized_title = f"Table_{i + 1}"  # Default title if no mapping is provided

        filename = os.path.join(folder_path, f"{sanitized_title}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved table {i + 1} as '{sanitized_title}' to {filename}")

if __name__ == "__main__":
    # URLs for squad and player data
    squad_url = "https://fbref.com/en/comps/9/Premier-League-Stats"
    player_url = "https://fbref.com/en/comps/9/stats/Premier-League-Stats#all_stats_standard"

    # Titles for squad tables
    squad_table_titles = [
        "Premier League Overall",
        "Premier League Home/Away",
        "Squad Standard Stats",
        "Squad Standard Stats Opponent Stats",
        "Squad Goalkeeping",
        "Squad Goalkeeping Opponent Stats",
        "Squad Advanced Goalkeeping",
        "Squad Advanced Goalkeeping Opponent Stats",
        "Squad Shooting",
        "Squad Shooting Opponent Stats",
        "Squad Passing",
        "Squad Passing Opponent Stats",
        "Squad Pass Types",
        "Squad Pass Types Opponent Stats",
        "Squad Goal and Shot Creation",
        "Squad Goal and Shot Creation Opponent Stats",
        "Squad Defensive Actions",
        "Squad Defensive Actions Opponent Stats",
        "Squad Possession",
        "Squad Possession Opponent Stats",
        "Squad Playing Time",
        "Squad Playing Time Opponent Stats",
        "Squad Miscellaneous Stats",
        "Squad Miscellaneous Stats Opponent Stats"
    ]

    # Scrape and save squad data
    print("Scraping squad data...")
    squad_tables = scrape_tables(squad_url)
    save_tables(squad_tables, folder_path="fbref_data/squad_data", table_titles=squad_table_titles)

    # Scrape and save player data
    print("Scraping player data...")
    player_tables = scrape_tables(player_url)
    save_tables(player_tables, folder_path="fbref_data/player_data")