"""Streamlit interface for GroundedOmni-Assistant."""

from __future__ import annotations

import base64
import io
import os
from typing import Any

import streamlit as st
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from PIL import Image

from retrieval import build_retriever
from workflow import build_graph


def render_evidence(evidence: list[dict[str, Any]]) -> None:
    """Render source snippets, Markdown tables, and original figure crops."""
    if not evidence:
        return
    with st.expander("PDF evidence", expanded=False):
        for item in evidence:
            st.caption(f"{item.get('source', 'PDF')} - page {item.get('page', '?')} - {item.get('kind', 'text')}")
            if item.get("kind") == "image" and item.get("image_data"):
                try:
                    image_bytes = base64.b64decode(item["image_data"])
                    st.image(
                        Image.open(io.BytesIO(image_bytes)),
                        caption=item.get("summary", "PDF figure"),
                        use_container_width=True,
                    )
                except Exception:
                    st.write(item.get("summary", "Figure unavailable."))
            elif item.get("kind") == "table":
                st.markdown(item.get("page_content", ""))
                if item.get("summary"):
                    st.caption(item["summary"])
            else:
                st.markdown(item.get("page_content", ""))


def main() -> None:
    st.set_page_config(page_title="GroundedOmni-Assistant", page_icon="page_facing_up", layout="wide")
    st.title("GroundedOmni-Assistant")
    st.caption("Ask questions grounded in the text, tables, and figures in your PDFs.")

    with st.sidebar:
        st.header("Documents & model")
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        api_key = st.text_input("OpenAI API key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
        model_name = st.text_input("Vision-capable chat model", value="gpt-4o-mini")
        top_k = st.slider("Retrieved evidence items", min_value=2, max_value=12, value=6)
        if st.button("Index PDFs", type="primary", disabled=not uploads or not api_key):
            with st.spinner("Extracting and indexing PDF evidence..."):
                try:
                    chat_model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0)
                    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
                    retriever = build_retriever(uploads, chat_model, embeddings)
                    st.session_state.rag_graph = build_graph(retriever, chat_model, top_k)
                    st.session_state.rag_retriever = retriever
                    st.session_state.indexed_files = [upload.name for upload in uploads]
                    st.session_state.chat_history = []
                    st.success(f"Indexed {len(uploads)} PDF(s).")
                except Exception as error:
                    st.error(f"Could not index PDFs: {error}")

    if st.session_state.get("indexed_files"):
        st.caption("Indexed: " + ", ".join(st.session_state.indexed_files))
    for turn in st.session_state.get("chat_history", []):
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant":
                render_evidence(turn.get("evidence", []))

    question = st.chat_input(
        "Ask about the uploaded PDFs",
        disabled=not st.session_state.get("rag_graph"),
    )
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching the uploaded PDFs..."):
                try:
                    prior_messages: list[BaseMessage] = []
                    for turn in st.session_state.chat_history[:-1][-8:]:
                        message = HumanMessage if turn["role"] == "user" else AIMessage
                        prior_messages.append(message(content=turn["content"]))
                    st.session_state.rag_retriever.search_kwargs = {"k": top_k}
                    result = st.session_state.rag_graph.invoke(
                        {"question": question, "messages": prior_messages}
                    )
                    answer = result["answer"]
                    evidence = [
                        {"page_content": doc.page_content, **doc.metadata}
                        for doc in result.get("context", [])
                    ]
                except Exception as error:
                    answer = f"The PDF search failed: {error}"
                    evidence = []
            st.markdown(answer)
            render_evidence(evidence)
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "evidence": evidence}
        )