from typing import Literal

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import TokenTextSplitter


ChunkingStrategy = Literal["fixed", "overlap", "semantic"]


def _token_split(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return [chunk for chunk in chunks if chunk.page_content.strip()]


def split_documents(
    documents: list[Document], strategy: ChunkingStrategy, embedding_model
) -> list[Document]:
    if strategy == "fixed":
        return _token_split(documents, chunk_size=500, chunk_overlap=0)

    if strategy == "overlap":
        return _token_split(documents, chunk_size=500, chunk_overlap=125)

    if strategy == "semantic":
        splitter = SemanticChunker(
            embeddings=embedding_model,
            breakpoint_threshold_type="percentile",
        )
        chunks = splitter.split_documents(documents)
        return [chunk for chunk in chunks if chunk.page_content.strip()]

    raise ValueError(f"Unknown strategy: {strategy}")
