from bs4 import BeautifulSoup as soup
import requests
import pandas as pd
import time
import random
import os
from functools import reduce
import sys
from urllib.error import HTTPError, URLError
from datetime import datetime

# Ensure the data directory exists
DATA_DIR = '../data/'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Conservative rate limiting settings
REQUESTS_PER_MINUTE_LIMIT = 5  # Extremely low to avoid rate limits
BASE_WAIT_TIME = 10  # Base wait time in seconds
MAX_WAIT_TIME = 15  # Max wait time in seconds
LEAGUE = "Premier-League"
LEAGUE_ID = "9"

current_year = datetime.now().year
LATEST_SEASON = f"{current_year-1}-{current_year}"
PREVIOUS_SEASON = f"{current_year-2}-{current_year-1}"
SEASONS = [PREVIOUS_SEASON, LATEST_SEASON]

request_count = 0
start_time = time.time()


def enforce_rate_limit():
    """Enforce a conservative delay to avoid being blocked."""
    global request_count, start_time
    request_count += 1

    if request_count >= REQUESTS_PER_MINUTE_LIMIT:
        elapsed_time = time.time() - start_time
        if elapsed_time < 60:
            wait_time = 60 - elapsed_time
            print(f"Rate limit threshold reached. Pausing for {int(wait_time)} seconds...")
            time.sleep(wait_time)

        request_count = 0
        start_time = time.time()
    
    sleep_time = BASE_WAIT_TIME + random.uniform(0, MAX_WAIT_TIME - BASE_WAIT_TIME)
    print(f"Sleeping for {int(sleep_time)} seconds before next request...")
    time.sleep(sleep_time)


def fetch_with_retries(url):
    """Fetches a URL with retries and conservative rate limiting."""
    retries = 5  # Increase retries to handle rate limits better
    for attempt in range(retries):
        enforce_rate_limit()
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                wait_time = random.randint(180, 300)  # Wait 3-5 minutes
                print(f"HTTP 429: Rate limit exceeded. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
            elif attempt < retries - 1:
                print(f"HTTP error: {e}. Retrying in {BASE_WAIT_TIME} seconds...")
                time.sleep(BASE_WAIT_TIME)
            else:
                raise e
        except (requests.exceptions.RequestException, URLError) as e:
            print(f"Network error: {e}. Retrying in {BASE_WAIT_TIME} seconds...")
            time.sleep(BASE_WAIT_TIME)
    return None


def get_fixture_data(season):
    """Fetches fixture data for a season."""
    print(f'Getting fixture data for {season}...')
    url = f'https://fbref.com/en/comps/{LEAGUE_ID}/{season}/schedule/{season}-{LEAGUE}-Scores-and-Fixtures'
    
    response = fetch_with_retries(url)
    if response is None:
        print(f"Failed to fetch fixture data for {season}")
        return
    
    try:
        tables = pd.read_html(response.text)
        fixtures = tables[0][['Wk', 'Day', 'Date', 'Time', 'Home', 'Away', 'xG', 'xG.1', 'Score']].dropna()
        fixtures['season'] = season
        fixtures["game_id"] = fixtures.index

        file_path = os.path.join(DATA_DIR, f'premier_league_{season}_fixture_data.csv')
        fixtures.to_csv(file_path, header=True, index=False)
        print(f'Fixture data saved to {file_path}')
    except Exception as e:
        print(f'Error processing fixture data for {season}: {e}')


def get_match_links(season):
    """Extracts match links for a given season."""
    print(f'Getting match links for {season}...')
    url = f'https://fbref.com/en/comps/{LEAGUE_ID}/{season}/schedule/{season}-{LEAGUE}-Scores-and-Fixtures'

    response = fetch_with_retries(url)
    if response is None:
        print(f"Failed to fetch match links for {season}")
        return []

    match_links = []
    try:
        links = soup(response.text, "html.parser").find_all('a')
        for l in links:
            href = l.get('href', '')
            if '/en/matches/' in href and LEAGUE in href:
                full_url = 'https://fbref.com' + href
                if full_url not in match_links:
                    match_links.append(full_url)
    except Exception as e:
        print(f'Error parsing match links for {season}: {e}')
    
    return match_links


def player_data(match_links, season):
    """Scrapes player data for all matches in a given season."""
    print(f'Scraping player data for {season}...')
    player_data = pd.DataFrame([])

    for count, link in enumerate(match_links):
        response = fetch_with_retries(link)
        if response is None:
            print(f"Skipping {link} due to repeated errors.")
            continue

        try:
            tables = pd.read_html(response.text)
            for table in tables:
                try:
                    table.columns = table.columns.droplevel()
                except Exception:
                    continue

            def get_team_data(t1_idx, t2_idx, home):
                df = reduce(lambda left, right: pd.merge(left, right, 
                    on=['Player', 'Nation', 'Age', 'Min'], how='outer'), 
                    [tables[t1_idx], tables[t2_idx]]
                ).iloc[:-1]
                return df.assign(home=home, game_id=count)

            t1 = get_team_data(3, 9, 1)
            t2 = get_team_data(10, 16, 0)
            player_data = pd.concat([player_data, pd.concat([t1, t2]).reset_index()])

            print(f'{count+1}/{len(match_links)} matches collected')
            file_path = os.path.join(DATA_DIR, f'premier_league_{season}_player_data.csv')
            player_data.to_csv(file_path, header=True, index=False)

        except Exception as e:
            print(f'Error processing match {link}: {e}')


def main():
    for season in SEASONS:
        get_fixture_data(season)
        match_links = get_match_links(season)
        player_data(match_links, season)
    print("Data collection complete!")


if __name__ == '__main__':
    try:
        main()
    except HTTPError:
        print('The website refused access, try again later')
        time.sleep(5)
