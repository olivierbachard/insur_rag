import argparse
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama
from tqdm import tqdm
from evals.answer_evaluation import evaluate_answer
from evals.retrieval_evaluation import evaluate_retrieval
from evals.load_tests import load_tests
from evals.models import AnswerEval, RetrievalEval, TestQuestion
from src.rag_app import RAGApp
from src.vector_store import create_vector_store

from dotenv import load_dotenv

RETRIEVAL_K = 10


def run_all_retrieval_evaluations(tests: list[TestQuestion], retriever: VectorStoreRetriever):
    print("Starting retrieval evaluations...")
    results: list[RetrievalEval] = []
    total_tests = len(tests)

    for idx, test in tqdm(enumerate(tests)):
        chunks = retriever.invoke(test.question, k=RETRIEVAL_K)
        chunks_str = [chunk.page_content for chunk in chunks]
        result = evaluate_retrieval(test_question=test, chunks=chunks_str, k=RETRIEVAL_K)
        results.append(result)

    avg_mrr = sum(r.mrr for r in results) / total_tests
    avg_ndcg = sum(r.ndcg for r in results) / total_tests
    avg_coverage = sum(r.keywords_found_percentage for r in results) / total_tests

    print(f"Evaluation completed on {total_tests} tests.")
    print(f"Average MRR: {avg_mrr:.4f}")
    print(f"Average nDCG: {avg_ndcg:.4f}")
    print(f"Average Keyword Coverage: {avg_coverage:.2f}%")


def run_all_answer_evaluations(tests: list[TestQuestion], app: RAGApp):
    print("Starting answer evaluations...")
    
    results: list[tuple[AnswerEval, str, list[Document]]] = []
    for idx, test in tqdm(enumerate(tests)):

        result = evaluate_answer(test=test, rag_app=app)
        results.append(result)

    avg_accuracy = sum(r[0].accuracy for r in results) / len(results)
    avg_completeness = sum(r[0].completeness for r in results) / len(results)
    avg_relevance = sum(r[0].relevance for r in results) / len(results)

    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Completeness: {avg_completeness:.4f}")
    print(f"Average Relevance: {avg_relevance:.4f}")


def main(test_type: str):
    print("Starting evaluation of RAG system...")
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
    retriever = vector_store.as_retriever()

    tests = load_tests()

    if test_type in ["retrieval", "all"]:
        run_all_retrieval_evaluations(tests=tests, retriever=retriever)
    if test_type in ["answer", "all"]:
        # llm = ChatGoogleGenerativeAI(
        #         model="gemini-3.1-flash-lite-preview", max_tokens=512, timeout=None, max_retries=2
        #     )
        llm = ChatOllama(
            model="llama3.2", num_predict=512, validate_model_on_init=True
        )
        rag_app = RAGApp(
            retriever=retriever,
            llm=llm,
        )
        run_all_answer_evaluations(tests=tests[:3], app=rag_app)

    print("Evaluation completed.")


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["retrieval", "answer", "all"],
        default="retrieval",
        help="type of evaluation to run",
    )
    args = parser.parse_args()

    main(args.type)
