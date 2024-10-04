# football-analytics-RAG

This intention of this project is to create a rag application for technical analysis of football.

The user should be able to answer questions such as:
1. "Why is Rodri so important in Manchester City's midfield?"
2. "What risks are there to Ange Postecoglu's system?"
3. "What is Phil Foden's best position?"

## Installing requirements
From project directory run: pip install -r requirements.txt

## Open AI
I have built this to work with openAI, for the project to run you will need an api key.

When you start the project save a file called '.env'

This will need to contain your api key like this:
OPENAI_API_KEY='your_openai_api_key here'

## Getting data
Initial plan is to use langchain to process speech to text documents from youtube videos

to get a list of urls from a youtube playlist run the following command:
python get_playlist_urls.py https://www.youtube.com/your-playlist-here

urls, video titles and author will be saved to data folder in url_lists folder. 

Function also checks for duplicate entries, so can add the same playlist to get new vids.

My suggestions for this are:
- Adam Clery from four-four-two for tactical analysis
- Barnaby slater for Premier League transfer news
- The Athletic FC for everything

### Getting transcripts
We then need to get the content of the videos in the playlist.

To get transcripts run the following command:
python get_youtube_transcripts.py --input data/url_lists/playlist_urls.csv --output data/url_lists/transcripts.csv

### Creating vector store and embeddings
We now need to make the transcript data useful to an LLM, to do this run this command:
python create_vector_store_from_csv.py --csv_file data/url_lists/transcripts.csv --vectorstore_dir docs/chroma/ --embedding_model OpenAI

## Quenstion and Answering module
I have added a question and answering module, to use this add this to your python script:

from qa_module import ask_question

## Exploration
We can then start asking questions and getting answers, with 'ask_question'imported as above we get get answers to questions like this:

#### Code:
query = "What risks are there to Ange Postecoglu's system?"
answer = ask_question(query, vectorstore_dir="docs/chroma/", k=20)
print(answer)

#### Answer:
"The high intensity and pressing style of play in Ange Postecoglu's system can lead to players getting injured."

## Running as an app
We can run this locally from browser by running the following code:
streamlit run streamlit_app.py

