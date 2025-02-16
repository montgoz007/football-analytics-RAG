import os
import warnings
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

# Suppress specific LangChainDeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables (OpenAI API Key)
load_dotenv()

def ask_question(query, vectorstore_dir="docs/chroma/", k=4):
    """
    Ask a question and get a concise answer using the Chroma vector store and OpenAI LLM.

    Args:
        query (str): The question to ask.
        vectorstore_dir (str): Directory where the Chroma vector store is persisted.
        k (int): Number of documents to retrieve (default: 4).

    Returns:
        str: Final answer to the query.
    """

    # Load the OpenAI API key from the environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in the environment as 'OPENAI_API_KEY'.")

    # Load the Chroma vector store
    vectordb = Chroma(
        persist_directory=vectorstore_dir,
        embedding_function=OpenAIEmbeddings(openai_api_key=openai_api_key)
    )

    # Initialize OpenAI LLM
    llm = OpenAI(api_key=openai_api_key)

    # Create a RetrievalQA chain (retriever + LLM)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",  # or "refine"
        retriever=vectordb.as_retriever(search_kwargs={"k": k})
    )

    # Run the query and return the final answer
    return qa_chain.run(query)
