from langchain_ollama import ChatOllama


def get_llm(base_url: str, model: str) -> ChatOllama:
    return ChatOllama(base_url=base_url, model=model, temperature=0.1)


def generate_answer(llm: ChatOllama, query: str, contexts: list[str]) -> str:
    context_text = "\n\n---\n\n".join(contexts)
    prompt = (
        "You are a helpful assistant. Answer using only the given context.\n"
        "If the context is not enough, say you don't have enough information.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_text}\n\n"
        "Answer:"
    )
    response = llm.invoke(prompt)
    return response.content if isinstance(response.content, str) else str(response.content)
