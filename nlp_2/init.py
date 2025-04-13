# %%
import os
import time
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_pinecone import PineconeVectorStore

# Load environment variables from a .env file
load_dotenv()

# Store environment variables in separate variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
namespace = os.environ.get('PINECONE_NAMESPACE')
index_name = os.environ.get('PINECONE_INDEX_NAME')
# %%
# Read a single document
def read_doc(file_path):
    """
    Reads a document from the specified file path using PyPDFLoader.

    Args:
        file_path (str): The path to the PDF file to be loaded.

    Returns:
        list: A list of pages from the loaded document.

    Prints:
        str: A message indicating the number of pages in the loaded document.
    """
    file_loader = PyPDFLoader(file_path)
    document = file_loader.load()
    print(f"Loaded document with {len(document)} pages")
    return document


file_path = r'CV - Alex Barria.pdf'
total = read_doc(file_path)

# %%
# Split the document into chunks
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

documents = chunk_data(docs=total, chunk_size=300, chunk_overlap=50)
type(documents)

# %%
# Connect to Pinecone DB and manage index
pc = Pinecone(api_key=PINECONE_API_KEY)
spec = ServerlessSpec(cloud=cloud, region=region)

if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)
    print("Index {} deleted".format(index_name))

# Check if index already exists (it shouldn't if this is the first time)
if index_name not in pc.list_indexes().names():
    print("Index created with the name: {}".format(index_name))
    pc.create_index(
        index_name,
        dimension=512,  # dimensionality of text-embedding models/embedding-001
        metric='cosine',
        spec=spec
    )
else:
    print("Index with the name {} already exists".format(index_name))

# %%
# Load embedding model
from utils import SentenceTransformerWrapper

# Load the sentence transformer model
raw_model = SentenceTransformer("jinaai/jina-embeddings-v2-small-en", trust_remote_code=True)

# Wrap it for LangChain compatibility
embed_model = SentenceTransformerWrapper(raw_model)

# %%
# Create and upsert embeddings into Pinecone
docsearch = PineconeVectorStore.from_documents(
    documents=documents,
    index_name=index_name,
    embedding=embed_model,
    namespace=namespace
)

print("Upserted values to {} index".format(index_name))

time.sleep(1)

# %%
# RETRIEVE AND SEARCH INTO THE CREATED PINECONE DATABASES
vectorstore_cv = PineconeVectorStore(
    index_name=index_name,
    embedding=embed_model,
    namespace=namespace,
)

retriever_cv = vectorstore_cv.as_retriever()
# %%
query = "in which companies did alex used to work"
vectorstore_cv.similarity_search(query, k=1)
# %%
from langchain_groq import ChatGroq

chat = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",  # Or "llama3-8b-8192", etc.
    temperature=0,
    streaming=True
)

# %%
from langchain.chains import RetrievalQA  

query = "Give me a 1 paragraph summary about Alex"

qa_cv = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=vectorstore_cv.as_retriever()
)
result = qa_cv.invoke(query)

print(result['result'])
# %%
