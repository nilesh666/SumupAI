import sys, types
sys.modules['torch.classes'] = types.SimpleNamespace(__path__=[])

import streamlit as st
from generate import gate #, valid_api
from retrieve import get_coll
from qdrant_client import QdrantClient, models
from langchain_qdrant import Qdrant
from dotenv import load_dotenv
from retrieve import embed
import os
from load import insert
from load import load_pdfs
from retrieve import text_split

load_dotenv()

url = os.getenv("QDRANT_URL")

st.set_page_config(
    page_title = "Sum Up AI",
    page_icon = "🤖",
)

client = QdrantClient(
    url = url,
    prefer_grpc = False
)

st.title("Summarizer")

st.sidebar.subheader("Cohere API key")

def new_collection(c_name, ct):
    ct.create_collection(
        collection_name = c_name,
        vectors_config = models.VectorParams(size = 100, distance = models.Distance.COSINE),

    )


if not client.collection_exists(collection_name="test0"):
    new_collection("test0", client)

user_cohere = st.sidebar.text_input("Enter Key", type = "password")

st.sidebar.subheader("Add a new collection")
c_n = st.sidebar.text_input("Enter a collection name")
add = st.sidebar.button("Create Collection")
if add:
    new_collection(c_n, client)

options = get_coll(client)
st.sidebar.subheader("Select a collection")
selected = st.sidebar.selectbox("Choose one from the available collections", options = options)

st.sidebar.subheader("Delete a collection")
d_c = st.sidebar.text_input("Enter the name of the collection to delete")
delete = st.sidebar.button("Delete Collection")
if delete:
    if d_c in options:
        client.delete_collection(collection_name=d_c)
    else:
        st.sidebar.write("Oops!! Collection not found")

embeddings = embed()

db = Qdrant.from_existing_collection(
    embedding = embeddings,
    collection_name = selected,
    url = url
)

st.subheader("PDF upload section")

uploaded_files = st.file_uploader("Upload pdf files only", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    if st.button("Ingest"):
        doc = load_pdfs(uploaded_files)
        t = text_split(doc)
        st.write(f"Ingesting into {selected}.......")
        insert(selected, db, t, embeddings, url)
        st.write("Ingested")

# if st.button("Summarize"):
#     if client.count(collection_name = selected, exact=True):
#         if user_cohere:
#             query = "Summarize the documents highlighting all the metrics"
#             st.subheader("Generated Answer: ")
#             st.write(gate(query, db).content)
#
#         else:
#             st.write("Oops!! Empty collection")
#     else:
#         st.write("Oops!!....Please provide an api key to proceed")

query = "Summarize the documents highlighting all the metrics"

if st.button("Summarize"):
    if not user_cohere:
        st.warning("Please provide an api key")
    else:
        doc_cnt = client.count(collection_name = selected, exact=True)
        if doc_cnt and doc_cnt.count > 0:
            st.subheader("Generated Answer: ")
            with st.spinner("Generating"):
                resp = gate(query, db, str(user_cohere))
            st.write(resp.content)
        else:
            st.warning("Oops! The selected collection is empty")
