
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langchain_core.vectorstores import VectorStoreRetriever


class RAGApp:

    SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

    def __init__(self, retriever: VectorStoreRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm

    def answer_question(self, question: str, history: list[dict] = []) -> str:

        # improve context retrieval by including the conversation history in the query to the vector store
        history_contents = "\n".join([
            item["text"] for h in history for item in h["content"] if h["role"] == "user"
        ])
        
        documents = self.retriever.invoke(history_contents + "\n" + question)
        context = "\n\n".join([doc.page_content for doc in documents])
        
        system_prompt = self.SYSTEM_PROMPT.format(context=context)
        messages = [SystemMessage(content=system_prompt)]
        messages.extend(convert_to_messages(history))
        messages.append(HumanMessage(content=question))
        response = self.llm.invoke(messages)
        return response.content