import os
import re

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

url = os.getenv("QDRANT_URL")

def embed():
    model_name = "BAAI/bge-large-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def text_split(doc):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    texts = splitter.split_documents(doc)
    return texts

def get_coll(client):
    collections = client.get_collections()
    collections = str(collections)
    l = re.findall(r"name='(.*?)'", collections)
    return l
