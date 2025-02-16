import os
import warnings
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def ask_question(query, vectorstore_dir="docs/chroma/", k=4, model_path = "models/mistral-7b-v0.1.Q4_K_M.gguf"):
    """
    Ask a question using a local Llama model and ChromaDB.

    Args:
        query (str): The question to ask.
        vectorstore_dir (str): Directory where the Chroma vector store is stored.
        k (int): Number of documents to retrieve.
        model_path (str): Path to the local model.

    Returns:
        str: Answer to the query.
    """

    # Load local embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load the Chroma vector store
    vectordb = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)

    # Load the local LLaMA model
    llm = LlamaCpp(model_path=model_path, n_ctx=2048, verbose=False)

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "map_reduce" might be too heavy for local models
        retriever=vectordb.as_retriever(search_kwargs={"k": k})
    )

    # Run the query and return the final answer
    return qa_chain.run(query)
