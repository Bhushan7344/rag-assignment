try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # fallback for older installs
    from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str, local_files_only: bool = False
) -> HuggingFaceEmbeddings:
    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"local_files_only": local_files_only},
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load embedding model. If you are behind a proxy or offline, "
            "download 'sentence-transformers/all-MiniLM-L6-v2' first and set "
            "EMBEDDING_LOCAL_FILES_ONLY=true in .env."
        ) from exc
