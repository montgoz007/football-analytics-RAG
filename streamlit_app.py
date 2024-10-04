import os
import streamlit as st
from dotenv import load_dotenv
from qa_module import ask_question  # Import the ask_question function from qa_module.py

# Load environment variables from .env file
load_dotenv()

# Streamlit app layout
st.title("Football Guru App!!!")
st.write("Ask a question about football tactics, transfers and news")

# Input box for the user question
user_question = st.text_input("Your Question:")

if st.button("Ask"):
    if user_question:
        with st.spinner("Getting answer..."):
            # Get the answer using the existing functionality
            answer = ask_question(user_question)
            # Display the answer
            st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a question.")

