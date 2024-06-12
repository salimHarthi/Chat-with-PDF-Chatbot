import os 
from langchain_community.document_loaders import DirectoryLoader, PDFMinerLoader, PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from constants import CHROMA_SETTINGS

persist_directory = "db"

def main():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PyPDFLoader(os.path.join(root, file))
                documents = loader.load()
                print("splitting into chunks")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                texts = text_splitter.split_documents(documents)
                #create embeddings here
                print("Loading sentence transformers model")
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                #create vector store here
                print(f"Creating embeddings. May take some minutes...")
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
                db.persist()
                db=None 

    print(f"Ingestion complete!")

if __name__ == "__main__":
    main()