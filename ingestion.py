from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


def load_documents(source_path: Path) -> list[Document]:
    if not source_path.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".pdf":
        loader = PyPDFLoader(str(source_path))
        documents = loader.load()
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(source_path), encoding="utf-8")
        documents = loader.load()
    else:
        raise ValueError("Only PDF, TXT, and MD files are supported.")

    non_empty_docs = [doc for doc in documents if doc.page_content.strip()]
    if not non_empty_docs:
        raise ValueError(
            f"Source document is empty or has no extractable text: {source_path}"
        )

    return non_empty_docs
