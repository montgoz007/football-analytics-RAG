from bs4 import BeautifulSoup as soup
import requests
import pandas as pd
import time
import random
from functools import reduce
import sys
from urllib.error import HTTPError, URLError
from datetime import datetime

DATA_DIR = '../data/'
REQUESTS_PER_MINUTE_LIMIT = 9 # Number of requests per minute
BASE_WAIT_TIME = 3  # Base wait time in seconds
LEAGUE = "Premier-League"
LEAGUE_ID = "9"

current_year = datetime.now().year
LATEST_SEASON = f"{current_year-1}-{current_year}"
PREVIOUS_SEASON = f"{current_year-2}-{current_year-1}"
SEASONS = [PREVIOUS_SEASON, LATEST_SEASON]


def fetch_with_retries(url, first_request=False):
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                if first_request:
                    print("Rate limited on first request. Exiting program. Please wait before retrying.")
                    sys.exit(1)
                wait_time = 120  # 2 minutes wait time for 429 errors
                print(f"HTTP 429: Rate limit exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
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
    print(f'Getting fixture data for {season}...')
    url = f'https://fbref.com/en/comps/{LEAGUE_ID}/{season}/schedule/{season}-{LEAGUE}-Scores-and-Fixtures'
    
    response = fetch_with_retries(url, first_request=True)
    if response is None:
        print(f"Failed to fetch fixture data for {season}")
        return
    
    try:
        tables = pd.read_html(response.text)
        fixtures = tables[0][['Wk', 'Day', 'Date', 'Time', 'Home', 'Away', 'xG', 'xG.1', 'Score']].dropna()
        fixtures['season'] = season
        fixtures["game_id"] = fixtures.index
        
        file_path = f'{DATA_DIR}premier_league_{season}_fixture_data.csv'
        fixtures.to_csv(file_path, header=True, index=False)
        print(f'Fixture data saved to {file_path}')
    except Exception as e:
        print(f'Error processing fixture data for {season}: {e}')


def get_match_links(season):
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
    print(f'Scraping player data for {season}...')
    player_data = pd.DataFrame([])
    request_count = 0

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
            file_path = f'{DATA_DIR}premier_league_{season}_player_data.csv'
            player_data.to_csv(file_path, header=True, index=False)

            request_count += 1
            if request_count >= REQUESTS_PER_MINUTE_LIMIT:
                print("Rate limit reached. Pausing for 60 seconds...")
                time.sleep(60)
                request_count = 0
            else:
                time.sleep(BASE_WAIT_TIME + random.uniform(0, 2))
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
