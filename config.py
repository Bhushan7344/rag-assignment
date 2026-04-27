from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass
class Settings:
    base_dir: Path = Path(__file__).resolve().parent
    source_document: Path = base_dir / os.getenv("SOURCE_DOCUMENT", "sample.pdf")
    chroma_base_dir: Path = base_dir / os.getenv("CHROMA_BASE_DIR", "./chroma_store")
    output_dir: Path = base_dir / os.getenv("OUTPUT_DIR", "./outputs")
    top_k: int = int(os.getenv("TOP_K", "4"))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    embedding_local_files_only: bool = (
        os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "false").lower() == "true"
    )
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "mistral")


def get_settings() -> Settings:
    settings = Settings()
    settings.chroma_base_dir.mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "retrieved_chunks").mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "final_answers").mkdir(parents=True, exist_ok=True)
    (settings.output_dir / "charts").mkdir(parents=True, exist_ok=True)
    return settings
