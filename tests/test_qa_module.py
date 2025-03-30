import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import pytest
from unittest.mock import patch, MagicMock
from chains.qa_module import ask_question

@pytest.fixture
def mock_env():
    """Mock the environment variable for OpenAI API key."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_api_key"}):
        yield

@patch("chains.qa_module.Chroma")
@patch("chains.qa_module.OpenAIEmbeddings")
@patch("chains.qa_module.OpenAI")
@patch("chains.qa_module.RetrievalQA")
def test_ask_question_success(mock_retrievalqa, mock_openai, mock_embeddings, mock_chroma, mock_env):
    """Test ask_question with mocked dependencies."""
    # Mock the Chroma vector store
    mock_vectordb = MagicMock()
    mock_chroma.return_value = mock_vectordb

    # Mock the OpenAI LLM
    mock_llm = MagicMock()
    mock_openai.return_value = mock_llm

    # Mock the RetrievalQA chain
    mock_qa_chain = MagicMock()
    mock_retrievalqa.from_chain_type.return_value = mock_qa_chain

    # Mock the QA chain's run method
    mock_qa_chain.run.return_value = "Mocked Answer"

    # Call the function
    query = "What is the score of the last match?"
    result = ask_question(query)

    # Assertions
    mock_chroma.assert_called_once_with(
        persist_directory="docs/chroma/",
        embedding_function=mock_embeddings(openai_api_key="test_api_key")
    )
    mock_openai.assert_called_once_with(api_key="test_api_key")
    mock_retrievalqa.from_chain_type.assert_called_once_with(
        llm=mock_llm,
        chain_type="map_reduce",
        retriever=mock_vectordb.as_retriever(search_kwargs={"k": 4})
    )
    mock_qa_chain.run.assert_called_once_with(query)
    assert result == "Mocked Answer"

def test_ask_question_no_api_key():
    """Test ask_question when the API key is missing."""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="OpenAI API key not found. Please set it in the environment as 'OPENAI_API_KEY'."):
            ask_question("Test query")