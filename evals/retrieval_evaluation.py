import math


from evals.models import RetrievalEval, TestQuestion


def calculate_mrr(keyword: str, chunks: list[str]) -> float:
    keyword_lower = keyword.lower()
    for idx, chunk in enumerate(chunks, start=1):
        if keyword_lower in chunk.lower():
            return 1.0 / idx
    return 0.0


def calculate_dcg(relevances: list[int], k: int) -> float:
    dcg = 0.0
    for i in range(min(k, len(relevances))):
        dcg += relevances[i] / math.log2(i + 2)  # i+2 because rank starts at 1
    return dcg


def calculate_ndcg(keyword: str, chunks: list[str], k: int) -> float:
    keyword_lower = keyword.lower()
    relevances = [1 if keyword_lower in chunk.lower() else 0 for chunk in chunks[:k]]
    dcg = calculate_dcg(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = calculate_dcg(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_retrieval(
    test_question: TestQuestion, chunks: list[str], k: int = 10
) -> RetrievalEval:
    mrr = [calculate_mrr(keyword, chunks) for keyword in test_question.keywords]
    avg_mrr = sum(mrr) / len(mrr) if mrr else 0.0

    ndcg = [calculate_ndcg(keyword, chunks, k) for keyword in test_question.keywords]
    avg_ndcg = sum(ndcg) / len(ndcg) if ndcg else 0.0

    keywords_found = sum(1 for score in mrr if score > 0)
    total_keywords = len(test_question.keywords)
    keywords_found_percentage = (
        (keywords_found / total_keywords) * 100 if total_keywords > 0 else 0.0
    )

    return RetrievalEval(
        mrr=avg_mrr,
        ndcg=avg_ndcg,
        keywords_found=keywords_found,
        total_keywords=total_keywords,
        keywords_found_percentage=keywords_found_percentage,
    )
