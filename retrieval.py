"""Build the Chroma-backed multi-vector PDF retriever."""

from __future__ import annotations

import uuid
from typing import Any

from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from document_processing import extract_pdf_documents


def build_retriever(
    uploads: list[Any], llm: ChatOpenAI, embeddings: OpenAIEmbeddings
) -> MultiVectorRetriever:
    """Store summaries in Chroma and original evidence in an in-memory docstore."""
    vectorstore = Chroma(
        collection_name=f"groundedomni_{uuid.uuid4().hex}",
        embedding_function=embeddings,
    )
    docstore: InMemoryStore[Document] = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=docstore, id_key="doc_id")
    all_parents: list[Document] = []
    all_children: list[Document] = []
    for upload in uploads:
        parents, children = extract_pdf_documents(upload.name, upload.getvalue(), llm)
        all_parents.extend(parents)
        all_children.extend(children)
    if not all_children:
        raise ValueError("No searchable text, tables, or images could be extracted from these PDFs.")
    docstore.mset([(doc.metadata.pop("_doc_id"), doc) for doc in all_parents])
    vectorstore.add_documents(all_children)
    return retriever