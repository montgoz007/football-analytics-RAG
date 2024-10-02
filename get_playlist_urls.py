import sys
from pytube import Playlist

def get_playlist_urls(playlist_url, output_file=None):
    """Fetch and optionally save the URLs of all videos in a YouTube playlist."""
    playlist = Playlist(playlist_url)

    # Print or save the URLs
    if output_file:
        with open(output_file, 'w') as file:
            for url in playlist.video_urls:
                file.write(f"{url}\n")
        print(f"URLs have been saved to {output_file}")
    else:
        for url in playlist.video_urls:
            print(url)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_playlist_urls.py <playlist_url> [output_file]")
        sys.exit(1)

    playlist_url = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    get_playlist_urls(playlist_url, output_file)
