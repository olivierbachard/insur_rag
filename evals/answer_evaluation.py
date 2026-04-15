


from langchain_core.documents import Document
from litellm import completion

from evals.models import AnswerEval, TestQuestion
from src.rag_app import RAGApp

# LLM_JUDGE_MODEL = "gemini/gemini-3.1-flash-lite-preview"
# LLM_JUDGE_MODEL = "gemini/gemma-3-12b"
LLM_JUDGE_MODEL = "ollama/llama3.2"

LLM_JUDGE_SYSTEM_PROMPT = """
You are an expert judge evaluating the quality of an answer provided by a RAG system.
Evaluate the generated answer by comparing it to the reference answer. Only give 5/5 scores for perfect answers.
"""

LLM_JUDGE_USER_PROMPT = """
Question:
{question}

Generated Answer:
{generated_answer}

Reference Answer:
{reference_answer}

Please evaluate the generated answer on three dimensions:
1. Accuracy: How factually correct is it compared to the reference answer? Only give 5/5 scores for perfect answers.
2. Completeness: How thoroughly does it address all aspects of the question, covering all the information from the reference answer?
3. Relevance: How well does it directly answer the specific question asked, giving no additional information?

Provide detailed feedback and scores from 1 (very poor) to 5 (ideal) for each dimension. If the answer is wrong, then the accuracy score must be 1.
"""

def evaluate_answer(test: TestQuestion, rag_app: RAGApp) -> tuple[AnswerEval, str, list[Document]]:
    
    answer, documents = rag_app.answer_question(test.question)

    messages = [
        {
            "role": "system",
            "content": LLM_JUDGE_SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": LLM_JUDGE_USER_PROMPT.format(
                question=test.question,
                generated_answer=answer,
                reference_answer=test.reference_answer
            )
        }
    ]

    judgement_response = completion(model=LLM_JUDGE_MODEL, messages=messages, response_format=AnswerEval)
    print(judgement_response.choices[0].message.content)
    answer_eval = AnswerEval.model_validate_json(judgement_response.choices[0].message.content)
    return answer_eval, answer, documents

