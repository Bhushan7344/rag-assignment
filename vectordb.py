from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document


def build_vector_store(
    documents: list[Document],
    embeddings,
    persist_dir: Path,
    collection_name: str,
) -> Chroma:
    if not documents:
        raise ValueError(
            f"No chunks generated for collection '{collection_name}'. "
            "Check document content and chunking configuration."
        )

    collection_dir = persist_dir / collection_name
    collection_dir.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=str(collection_dir),
    )
