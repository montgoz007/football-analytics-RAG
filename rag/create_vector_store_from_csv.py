import os
import shutil
import argparse
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

def create_vector_store_from_csv(csv_file, vectorstore_dir, embedding_model="local"):
    """
    Create a vector store from transcripts in the specified CSV file.

    Args:
        csv_file (str): Path to the CSV file containing transcripts.
        vectorstore_dir (str): Directory to store the vector database.
        embedding_model (str): Embedding model to use (default: "local").
    """
    
    # Check if the vector store directory already exists
    if os.path.exists(vectorstore_dir):
        print(f"Vector store directory '{vectorstore_dir}' already exists.")
        user_input = input("Do you want to overwrite it? (yes/no): ").strip().lower()
        if user_input != "yes":
            print("Skipping vector store creation to avoid duplicates.")
            return
        else:
            # Remove the existing vector store if the user confirms
            shutil.rmtree(vectorstore_dir)
            print(f"Removed existing vector store at {vectorstore_dir}")

    # Load the CSV file containing URLs and transcripts
    print(f"Loading transcripts from CSV: {csv_file}")
    df = pd.read_csv(csv_file)

    # Check if the expected 'transcript' column exists
    if 'transcript' not in df.columns:
        raise ValueError(f"'transcript' column not found in CSV file {csv_file}")
    
    # Create Document objects from the transcripts in the CSV file
    docs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading transcripts", unit="row"):
        transcript = row['transcript']  # Assuming 'transcript' is the column containing the transcripts
        title = row['title'] if 'title' in df.columns else "No Title"  # Optional: use title as metadata
        docs.append(Document(page_content=transcript, metadata={"title": title}))

    print(f"Loaded {len(docs)} documents.")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, 
        chunk_overlap=50,
    )
    split_docs = text_splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks.")

    # Initialize embedding model
    if embedding_model == "local":
        # Use HuggingFace Embeddings with a local model like 'sentence-transformers/all-MiniLM-L6-v2'
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif embedding_model == "openai":
        # Use OpenAI Embeddings
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    else:
        raise ValueError(f"Embedding model '{embedding_model}' is not supported")

    # Create and persist the vector store with a progress bar
    print("Creating vector store and generating embeddings...")

    vectordb = Chroma.from_documents(
        documents=tqdm(split_docs, desc="Processing chunks", unit="chunk"),  # Progress bar for chunks
        embedding=embeddings,
        persist_directory=vectorstore_dir
    )

    print(f"Vector store created with {vectordb._collection.count()} documents.")
    print(f"Vector store saved to: {vectorstore_dir}")

if __name__ == "__main__":
    # Set up argument parser for command-line execution
    parser = argparse.ArgumentParser(description="Create a Chroma vector store from YouTube transcripts in a CSV file.")
    
    parser.add_argument(
        "--csv_file",
        type=str,
        default=os.path.join(os.getcwd(), "data/url_lists/transcripts.csv"),  # Changed to transcripts.csv
        help="Path to the CSV file containing transcripts (default: data/url_lists/transcripts.csv)"
    )
    parser.add_argument(
        "--vectorstore_dir",
        type=str,
        default="docs/chroma_local/",  # Updated directory
        help="Directory to save the vector store (default: docs/chroma_local/)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="local",
        help="Embedding model to use (default: local)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Call the function to create the vector store
    create_vector_store_from_csv(
        csv_file=args.csv_file,
        vectorstore_dir=args.vectorstore_dir,
        embedding_model=args.embedding_model
    )
