from pathlib import Path

from src.vector_store import create_vector_store

from dotenv import load_dotenv


def main():
    print("Hello from insur-rag!")
    knowledge_base_path = str(Path(__file__).parent / "knowledge_base")
    db_path = str(Path(__file__).parent / "chroma_db")
    print(f"Fetching documents from: {knowledge_base_path}")
    vectors_tore = create_vector_store(knowledge_base_path=knowledge_base_path, db_path=db_path)
    collection = vectors_tore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimension = len(sample_embedding)
    print(f"Vector store created with {count} documents and embedding dimension of {dimension}.")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
