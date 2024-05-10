import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

embed_model = SentenceTransformerEmbeddings(model_name="llmware/industry-bert-insurance-v0.1")
load1 = DirectoryLoader('data/', glob="**/*.pdf", show_progress=True, loader_cls=PyPDFLoader)
doc= load1.load()
splitText = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=70)
texts = splitText.split_documents(doc)

vector_store = Chroma.from_documents(texts, embed_model, collection_metadata={"hnsw:space": "cosine"}, persist_directory="operatingsystem/embed")
