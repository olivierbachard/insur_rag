import glob
import os
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma


def fetch_documents(knowledge_base_path: Path):
    folders = glob.glob(f"{knowledge_base_path}/*")
    all_documents = []
    for folder in folders:
        doc_type = Path(folder).name
        print(f"Document type: {doc_type}")
        # loader = DirectoryLoader(folder, glob="**/*.md")
        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        documents = loader.load()
        for doc in documents:
            doc.metadata["doc_type"] = doc_type
            all_documents.append(doc)

    return all_documents


def create_chunks(documents: list) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(
    knowledge_base_path: Path, db_path: Path, embeddings: Embeddings, force_recreate: bool = True
) -> VectorStore:
    
    if os.path.exists(db_path):
        if not force_recreate:
            return Chroma(persist_directory=db_path, embedding_function=embeddings)
        
        print("Deleting existing vector store.")
        Chroma(persist_directory=db_path, embedding_function=embeddings).delete_collection()

    documents = fetch_documents(knowledge_base_path)
    vector_store = Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=db_path
    )
    return vector_store
