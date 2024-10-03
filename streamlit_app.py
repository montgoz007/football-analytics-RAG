import os
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Function to create the vector store
def create_vector_store(vectorstore_dir):
    # Initialize the vector store
    embedding = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectordb = Chroma(persist_directory=vectorstore_dir, embedding_function=embedding)
    return vectordb

# Load vector store from existing directory
vectorstore_dir = "docs/chroma/"
vectordb = create_vector_store(vectorstore_dir)

# Initialize the LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set up the RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="map_reduce",
    retriever=vectordb.as_retriever()
)

# Streamlit app layout
st.title("RAG Question-Answering App")
st.write("Ask a question related to the transcripts!")

# Input box for the user question
user_question = st.text_input("Your Question:")

if st.button("Ask"):
    if user_question:
        with st.spinner("Getting answer..."):
            # Get the answer
            answer = qa_chain.run(user_question)
            # Display the answer
            st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a question.")
