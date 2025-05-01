import tempfile
import PyPDF2
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain.docstore.document import Document

load_dotenv()

def load_pdfs(uploaded_files):
    documents = []
    if uploaded_files:
        for i in uploaded_files:
            with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp_file:
                tmp_file.write(i.read())
                tmp_path = tmp_file.name
            reader = PdfReader(tmp_path)
            text = ""
            for pg in reader.pages:
                text += pg.extract_text() or ""

            dc = Document(page_content=text, metadata = {"source": i.name})
            documents.append(dc)
    return documents

def insert(c_name, db, t, e, u):
    client = QdrantClient(
        url = u,
        prefer_grpc = False
    )
    q = Qdrant.from_documents(
            t,
            e,
            url = u,
            prefer_grpc = False,
            collection_name = c_name,
            force_recreate = True
        )
