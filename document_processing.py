"""Extract and summarize text, tables, and figures from PDF documents."""

from __future__ import annotations

import base64
import io
import uuid
from typing import Any

import fitz
import pdfplumber
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def table_to_markdown(rows: list[list[Any]]) -> str:
    """Render a pdfplumber table as a compact Markdown table."""
    cleaned = [[str(cell or "").replace("\n", " ").strip() for cell in row] for row in rows]
    cleaned = [row for row in cleaned if any(row)]
    if not cleaned:
        return ""
    width = max(map(len, cleaned))
    cleaned = [row + [""] * (width - len(row)) for row in cleaned]
    lines = [
        "| " + " | ".join(cleaned[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in cleaned[1:])
    return "\n".join(lines)


def _summarize_table(llm: ChatOpenAI, table: str) -> str:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Describe the provided table briefly for retrieval. Only state details explicitly present in it."),
            ("human", "{table}"),
        ]
    )
    return str((prompt | llm).invoke({"table": table}).content).strip()


def _summarize_image(llm: ChatOpenAI, image_bytes: bytes, mime_type: str) -> str:
    """Ask a vision-capable chat model for a strictly visual description."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe only visible content, labels, and relationships. Do not infer unseen facts."},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ]
    )
    return str(llm.invoke([message]).content).strip()


def extract_pdf_documents(
    file_name: str, pdf_bytes: bytes, llm: ChatOpenAI
) -> tuple[list[Document], list[Document]]:
    """Return original parent documents and searchable child summaries."""
    parents: list[Document] = []
    children: list[Document] = []
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as plumber_pdf:
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                metadata = {"source": file_name, "page": page_number}
                extracted_page = plumber_pdf.pages[page_index]

                text = (extracted_page.extract_text() or "").strip()
                if text:
                    parent_id = str(uuid.uuid4())
                    parent = Document(page_content=text, metadata={**metadata, "kind": "text", "_doc_id": parent_id})
                    parents.append(parent)
                    children.append(
                        Document(page_content=text[:6000], metadata={**metadata, "doc_id": parent_id, "kind": "text"})
                    )

                for table_index, rows in enumerate(extracted_page.extract_tables() or []):
                    markdown = table_to_markdown(rows)
                    if not markdown:
                        continue
                    try:
                        summary = _summarize_table(llm, markdown)
                    except Exception:
                        summary = f"Table on page {page_number} of {file_name}."
                    parent_id = str(uuid.uuid4())
                    parents.append(
                        Document(
                            page_content=markdown,
                            metadata={
                                **metadata,
                                "kind": "table",
                                "summary": summary,
                                "table": table_index + 1,
                                "_doc_id": parent_id,
                            },
                        )
                    )
                    children.append(
                        Document(
                            page_content=summary + "\n" + markdown[:4000],
                            metadata={**metadata, "doc_id": parent_id, "kind": "table"},
                        )
                    )

                for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
                    if block.get("type") != 1 or not block.get("image"):
                        continue
                    image_bytes = block["image"]
                    if len(image_bytes) > 12_000_000:
                        continue
                    image_ext = str(block.get("ext", "png")).lower()
                    mime_type = "image/jpeg" if image_ext in {"jpg", "jpeg"} else f"image/{image_ext}"
                    try:
                        summary = _summarize_image(llm, image_bytes, mime_type)
                    except Exception:
                        summary = f"Figure on page {page_number} of {file_name}; image description unavailable."
                    parent_id = str(uuid.uuid4())
                    parents.append(
                        Document(
                            page_content=summary,
                            metadata={
                                **metadata,
                                "kind": "image",
                                "summary": summary,
                                "image_data": base64.b64encode(image_bytes).decode("ascii"),
                                "image_mime": mime_type,
                                "figure": block_index + 1,
                                "_doc_id": parent_id,
                            },
                        )
                    )
                    children.append(
                        Document(page_content=summary, metadata={**metadata, "doc_id": parent_id, "kind": "image"})
                    )
    finally:
        pdf.close()
    return parents, children