"""LangGraph retrieval and grounded-answer workflow."""

from __future__ import annotations

from typing import Any

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from schemas import AssistantState, NO_ANSWER, SYSTEM_PROMPT


def build_graph(retriever: MultiVectorRetriever, llm: ChatOpenAI, top_k: int) -> Any:
    """Compile retrieval and grounded response generation into a LangGraph workflow."""
    retriever.search_kwargs = {"k": top_k}

    def retrieve_context(state: AssistantState) -> dict[str, Any]:
        return {"context": retriever.invoke(state["question"])}

    def grounded_explanation_node(state: AssistantState) -> dict[str, str]:
        evidence = "\n\n".join(
            f"[{doc.metadata.get('source', 'PDF')}, page {doc.metadata.get('page', '?')}, "
            f"{doc.metadata.get('kind', 'text')}]\n"
            f"{doc.metadata.get('summary', doc.page_content)}\n"
            f"{doc.page_content if doc.metadata.get('kind') == 'table' else ''}"
            for doc in state.get("context", [])
        )
        if not evidence.strip():
            return {"answer": NO_ANSWER}
        history = state.get("messages", [])[-8:]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                *[(message.type, message.content) for message in history],
                ("human", "Question: {question}\n\nRetrieved PDF evidence:\n{evidence}"),
            ]
        )
        answer = (prompt | llm).invoke(
            {"question": state["question"], "evidence": evidence}
        ).content
        return {"answer": str(answer).strip()}

    workflow = StateGraph(AssistantState)
    workflow.add_node("retrieve_context", retrieve_context)
    workflow.add_node("grounded_explanation_node", grounded_explanation_node)
    workflow.add_edge(START, "retrieve_context")
    workflow.add_edge("retrieve_context", "grounded_explanation_node")
    workflow.add_edge("grounded_explanation_node", END)
    return workflow.compile()