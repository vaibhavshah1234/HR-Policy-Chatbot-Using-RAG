from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyPDFLoader
import os

def load_documents(data_path='data'):
    documents = []
    for file in os.listdir(data_path):
        if file.endswith('.pdf'):
            loader = PyPDFLoader(os.path.join(data_path, file))
            documents.extend(loader.load())
    return documents

def create_vectorstore():
    docs = load_documents()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vectorstore/faiss_index")

if __name__ == "__main__":
    create_vectorstore()
