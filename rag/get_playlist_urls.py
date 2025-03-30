import os
import sys
import pandas as pd
from pytube import Playlist, exceptions

def get_playlist_info(playlist_url, output_file):
    """Fetch and append the URLs, titles, YouTuber names, and playlist names of all videos in a YouTube playlist to a CSV file."""
    playlist = Playlist(playlist_url)
    playlist_name = playlist.title  # Get the playlist name
    
    # Ensure the 'data/url_lists' directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing data if the file exists
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        existing_data = set(zip(df_existing.get('url', []), df_existing.get('title', []), df_existing.get('youtuber', []), df_existing.get('playlist_name', [])))
    else:
        df_existing = pd.DataFrame(columns=['url', 'title', 'youtuber', 'playlist_name'])
        existing_data = set()
    
    # Prepare new data list
    new_data = []
    
    # Collect new URLs, titles, YouTuber names, and playlist names
    for video in playlist.videos:
        url = video.watch_url
        try:
            title = video.title
            youtuber = video.author  # Get the YouTuber's name (channel name)
        except exceptions.PytubeError as e:
            print(f"Error retrieving information for {url}: {e}")
            continue  # Skip this video if there's an error

        if (url, title, youtuber, playlist_name) not in existing_data:
            new_data.append({'url': url, 'title': title, 'youtuber': youtuber, 'playlist_name': playlist_name})
    
    # If there are new videos, append them
    if new_data:
        df_new = pd.DataFrame(new_data)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
        print(f"{len(new_data)} new videos have been added to {output_file}")
    else:
        print("No new videos to add.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_playlist_urls.py <playlist_url>")
        sys.exit(1)

    playlist_url = sys.argv[1]

    # Define the output file in 'data/url_lists'
    output_file = os.path.join('data', 'url_lists', 'playlist_urls.csv')

    get_playlist_info(playlist_url, output_file)
