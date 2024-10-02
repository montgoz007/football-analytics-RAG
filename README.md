# football-analytics-RAG

This intention of this project is to create a rag application for technical analysis of football.

The user should be able to answer questions such as:
1. "Why is Rodri so important in Manchester City's midfield?"
2. "What risks are there to Ange Postecoglu's system?"
3. "What is Phil Foden's best position?"

## Getting data
Initial plan is to use langchain to process speech to text documents from youtube videos

to get a list of urls from a youtube playlist run the following command:
python get_playlist_urls.py https://www.youtube.com/playlist?list=PLsLsr76bONJmnmSMQ5NmNLPEu8cXFXIwU "playlist_urls.txt"

urls will be saved to a .txt file.