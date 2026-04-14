from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from src.rag_app import RAGApp
from src.vector_store import create_vector_store
import gradio as gr

from dotenv import load_dotenv


def main():
    print("Hello from insur-rag!")
    knowledge_base_path = str(Path(__file__).parent / "knowledge_base")
    db_path = str(Path(__file__).parent / "chroma_db")
    print(f"Fetching documents from: {knowledge_base_path}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = create_vector_store(
        knowledge_base_path=knowledge_base_path,
        db_path=db_path,
        embeddings=embeddings,
        force_recreate=False,
    )
    collection = vector_store._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimension = len(sample_embedding)
    print(f"Vector store created with {count} documents and embedding dimension of {dimension}.")

    # model = HuggingFaceEndpoint(
    #         repo_id="openai/gpt-oss-120b",
    #         task="text-generation",
    #         max_new_tokens=512,
    #         do_sample=False,
    #     )
    
    # llm = ChatHuggingFace(llm=model, verbose=False)
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite-preview",
        max_tokens=512,
        timeout=None,
        max_retries=2,
    )

    app = RAGApp(retriever=vector_store.as_retriever(), llm=llm)

    gr.ChatInterface(fn=app.answer_question).launch()


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
