import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.schema import Document
import pypdf

load_dotenv()

folder = os.getenv("PDF_FOLDER_PATH")

def load_pdfs(folder_path):
    l = []
    for i in os.listdir(folder_path):
        if i.endswith('.pdf'):
            file_path = os.path.join(folder_path, i)
            reader = PdfReader(file_path)
            text = ""
            for pg in reader.pages:
                text += pg.extract_text() or ""
                docs = Document(page_content = text, metadata = {"source": i})
                l.append(docs)
    return l

doc = load_pdfs(folder)

txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 50
)

txts = txt_splitter.split_documents(doc)

model_name = "BAAI/bge-large-en"
model_kwargs = {'device':'cpu'}

encode_kwargs = {'normalize_embeddings':False}

print("Embedding started.................")

embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
)

print("Embedding Model Loaded................")

url = os.getenv("QDRANT_URL")
collection_name = "trial_db_2"

print("Pushing to Qdrant.................")

qdrant = Qdrant.from_documents(
    txts,
    embeddings,
    url = url,
    prefer_grpc = False,
    collection_name = collection_name
)

print("Qdrant index initialized...............")



