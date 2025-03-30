import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from tqdm import tqdm
import argparse
import os

def get_video_id(url):
    # Extract video ID from YouTube URL
    if 'watch?v=' in url:
        return url.split('watch?v=')[1].split('&')[0]  # Remove additional parameters if any
    elif 'youtu.be/' in url:
        return url.split('youtu.be/')[1].split('?')[0]  # Remove additional parameters if any
    return None

def get_transcript(video_id):
    try:
        # Fetch transcript using the API
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return ' '.join([item['text'] for item in transcript])
    except (NoTranscriptFound, TranscriptsDisabled):
        return 'Transcript not available'
    except Exception as e:
        return f"Error fetching transcript: {str(e)}"

def process_csv(input_path, output_path):
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return

    # Load the CSV file
    df = pd.read_csv(input_path)

    # Debugging: Check if 'url' column exists
    if 'url' not in df.columns:
        print(f"Error: The input CSV does not contain a 'url' column.")
        return

    # Add a new column with video IDs
    df['video_id'] = df['url'].apply(get_video_id)

    # Debugging: Check for missing or invalid video IDs
    invalid_ids = df[df['video_id'].isnull()]
    if not invalid_ids.empty:
        print(f"Warning: The following rows have invalid URLs and no video IDs:")
        print(invalid_ids[['url']])

    # Create a progress bar for processing transcripts
    tqdm.pandas(desc="Processing transcripts")

    # Fetch transcripts and store them in a new column, with a progress bar
    df['transcript'] = df['video_id'].progress_apply(lambda vid: get_transcript(vid) if vid else 'Invalid URL')

    # Save the updated DataFrame to a new CSV file
    df.to_csv(output_path, index=False)
    print(f"Transcripts saved successfully to {output_path}!")

if __name__ == "__main__":
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Fetch YouTube video transcripts and save them to a CSV file.")
    parser.add_argument('--input', required=True, help="Path to the input CSV file containing video URLs.")
    parser.add_argument('--output', required=True, help="Path to the output CSV file where transcripts will be saved.")

    args = parser.parse_args()

    # Process the CSV to fetch transcripts
    process_csv(args.input, args.output)
