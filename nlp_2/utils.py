from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class SentenceTransformerWrapper(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

def chunk_data(docs, chunk_size=100, chunk_overlap=20):
    """
    Splits a list of documents into smaller chunks using a recursive character text splitter.

    Args:
        docs (list): A list of documents to be split. Each document is expected to be a string.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 100.
        chunk_overlap (int, optional): The number of overlapping characters between consecutive chunks. Defaults to 20.

    Returns:
        list: A list of chunks obtained by splitting the input documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_splitter.split_documents(docs)
    return doc