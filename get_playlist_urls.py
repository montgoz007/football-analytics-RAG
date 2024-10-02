import os
import sys
import pandas as pd
from pytube import Playlist

def get_playlist_urls(playlist_url, output_file):
    """Fetch and append the URLs of all videos in a YouTube playlist to a CSV file."""
    playlist = Playlist(playlist_url)
    urls = playlist.video_urls
    
    # Ensure the 'data/url_lists' directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load existing URLs if the file exists, otherwise create an empty DataFrame
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        existing_urls = df_existing['url'].tolist()
    else:
        df_existing = pd.DataFrame(columns=['url'])
        existing_urls = []
    
    # Filter out duplicates
    new_urls = [url for url in urls if url not in existing_urls]
    
    # If there are new URLs, append them
    if new_urls:
        df_new = pd.DataFrame(new_urls, columns=['url'])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(output_file, index=False)
        print(f"{len(new_urls)} new URLs have been added to {output_file}")
    else:
        print("No new URLs to add.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_playlist_urls.py <playlist_url>")
        sys.exit(1)

    playlist_url = sys.argv[1]

    # Define the output file in 'data/url_lists'
    output_file = os.path.join('data', 'url_lists', 'playlist_urls.csv')

    get_playlist_urls(playlist_url, output_file)
