import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd
import re
import os
from datetime import datetime
from io import StringIO
import shutil  # For clearing existing data

def sanitize_filename(name):
    """Sanitize filenames to remove invalid characters."""
    return re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

def scrape_premier_league_tables(url):
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
        # Get the table title (from <caption> or nearby heading)
        caption = table.find('caption')
        title = caption.text.strip() if caption else "Unnamed_Table"

        # Sanitize the title for use as a filename
        sanitized_title = sanitize_filename(title)

        try:
            # Convert the HTML table to a DataFrame using StringIO
            df = pd.read_html(StringIO(str(table)), header=[0, 1])[0]  # Combine multi-row headers

            # Flatten multi-row headers into single row
            df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]

            dataframes.append((sanitized_title, df))
        except ValueError:
            print(f"Skipping table '{sanitized_title}' - No valid data found.")

    return dataframes

def save_tables_by_year(year, tables):
    """Save tables in a folder structure based on the year."""
    # Define the new directory structure relative to the parent of the 'scrapers' folder
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "fbref_data", "team_data"))
    output_dir = os.path.join(base_dir, str(year))
    
    # Clear the directory if it already exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Save each table to a CSV file in the year-specific folder
    for title, df in tables:
        filename = os.path.join(output_dir, f"{title}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved table to {filename}")

if __name__ == "__main__":
    # Adjust year logic for seasons running from August to May
    current_year = datetime.now().year
    latest_season = f"{current_year - 1}-{current_year}"
    previous_season = f"{current_year - 2}-{current_year - 1}"
    two_years_ago_season = f"{current_year - 3}-{current_year - 2}"
    seasons = [two_years_ago_season, previous_season, latest_season]

    # Base URL format
    base_url = "https://fbref.com/en/comps/9/{season}/{season}-Premier-League-Stats"

    # Scrape and save tables for each season
    for season in seasons:
        url = base_url.format(season=season)
        print(f"Scraping tables for {season}...")
        tables = scrape_premier_league_tables(url)
        save_tables_by_year(season.split('-')[0], tables)