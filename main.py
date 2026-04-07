from pathlib import Path

from src.vector_store import fetch_documents


def main():
    print("Hello from insur-rag!")
    knowledge_base_path = str(Path(__file__).parent / "knowledge_base")
    print(f"Fetching documents from: {knowledge_base_path}")
    all_text = fetch_documents(knowledge_base_path)
    print(f"Fetched {len(all_text)} characters from the knowledge base.")


if __name__ == "__main__":
    main()
