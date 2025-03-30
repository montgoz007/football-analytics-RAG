import os
import pytest
from unittest.mock import patch, MagicMock, mock_open
from rag.create_vector_store_from_csv import create_vector_store_from_csv

@pytest.fixture
def mock_env_vars():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test_openai_api_key"}):
        yield

@patch("rag.create_vector_store_from_csv.os.path.exists")
@patch("rag.create_vector_store_from_csv.shutil.rmtree")
@patch("rag.create_vector_store_from_csv.pd.read_csv")
@patch("rag.create_vector_store_from_csv.tqdm")
@patch("rag.create_vector_store_from_csv.Document")
@patch("rag.create_vector_store_from_csv.RecursiveCharacterTextSplitter")
@patch("rag.create_vector_store_from_csv.HuggingFaceEmbeddings")
@patch("rag.create_vector_store_from_csv.OpenAIEmbeddings")
@patch("rag.create_vector_store_from_csv.Chroma")
def test_create_vector_store_local_embeddings(
    mock_chroma,
    mock_openai_embeddings,
    mock_huggingface_embeddings,
    mock_text_splitter,
    mock_document,
    mock_tqdm,
    mock_read_csv,
    mock_rmtree,
    mock_exists,
    mock_env_vars,
):
    # Mock behaviors
    mock_exists.return_value = False
    mock_read_csv.return_value = MagicMock(columns=["transcript"], iterrows=lambda: iter([(0, {"transcript": "test"})]))
    mock_text_splitter.return_value.split_documents.return_value = ["chunk1", "chunk2"]
    mock_huggingface_embeddings.return_value = "mock_embeddings"
    mock_chroma.from_documents.return_value._collection.count.return_value = 2

    # Call the function
    create_vector_store_from_csv("test.csv", "test_dir", embedding_model="local")

    # Assertions
    mock_read_csv.assert_called_once_with("test.csv")
    mock_text_splitter.return_value.split_documents.assert_called_once()
    mock_huggingface_embeddings.assert_called_once_with(model_name="sentence-transformers/all-MiniLM-L6-v2")
    mock_chroma.from_documents.assert_called_once()

@patch("rag.create_vector_store_from_csv.os.path.exists")
@patch("rag.create_vector_store_from_csv.shutil.rmtree")
@patch("rag.create_vector_store_from_csv.pd.read_csv")
@patch("rag.create_vector_store_from_csv.tqdm")
@patch("rag.create_vector_store_from_csv.Document")
@patch("rag.create_vector_store_from_csv.RecursiveCharacterTextSplitter")
@patch("rag.create_vector_store_from_csv.HuggingFaceEmbeddings")
@patch("rag.create_vector_store_from_csv.OpenAIEmbeddings")
@patch("rag.create_vector_store_from_csv.Chroma")
def test_create_vector_store_openai_embeddings(
    mock_chroma,
    mock_openai_embeddings,
    mock_huggingface_embeddings,
    mock_text_splitter,
    mock_document,
    mock_tqdm,
    mock_read_csv,
    mock_rmtree,
    mock_exists,
    mock_env_vars,
):
    # Mock behaviors
    mock_exists.return_value = False
    mock_read_csv.return_value = MagicMock(columns=["transcript"], iterrows=lambda: iter([(0, {"transcript": "test"})]))
    mock_text_splitter.return_value.split_documents.return_value = ["chunk1", "chunk2"]
    mock_openai_embeddings.return_value = "mock_embeddings"
    mock_chroma.from_documents.return_value._collection.count.return_value = 2

    # Call the function
    create_vector_store_from_csv("test.csv", "test_dir", embedding_model="openai")

    # Assertions
    mock_read_csv.assert_called_once_with("test.csv")
    mock_text_splitter.return_value.split_documents.assert_called_once()
    mock_openai_embeddings.assert_called_once_with(openai_api_key="test_openai_api_key")
    mock_chroma.from_documents.assert_called_once()

@patch("rag.create_vector_store_from_csv.os.path.exists")
@patch("rag.create_vector_store_from_csv.input", return_value="no")
def test_skip_vector_store_creation(mock_input, mock_exists):
    # Mock behaviors
    mock_exists.return_value = True

    # Call the function
    create_vector_store_from_csv("test.csv", "test_dir", embedding_model="local")

    # Assertions
    mock_input.assert_called_once_with("Do you want to overwrite it? (yes/no): ")

@patch("rag.create_vector_store_from_csv.os.path.exists")
@patch("rag.create_vector_store_from_csv.shutil.rmtree")
@patch("rag.create_vector_store_from_csv.pd.read_csv")
def test_missing_transcript_column(mock_read_csv, mock_rmtree, mock_exists):
    # Mock behaviors
    mock_exists.return_value = False
    mock_read_csv.return_value = MagicMock(columns=["title"])

    # Call the function and assert exception
    with pytest.raises(ValueError, match="'transcript' column not found in CSV file test.csv"):
        create_vector_store_from_csv("test.csv", "test_dir", embedding_model="local")