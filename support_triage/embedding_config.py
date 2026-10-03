from support_triage.runtime import Credentials, setting

DEFAULT_MODELS = {"openai": "text-embedding-3-small", "local": "all-MiniLM-L6-v2"}


def resolve_embedding_provider(preferred=None, credentials=None):
    credentials = credentials if credentials is not None else Credentials.from_environment()
    preferred = preferred or setting("EMBEDDING_PROVIDER", "auto")
    if preferred == "auto":
        return "openai" if credentials.openai else "local"
    if preferred not in DEFAULT_MODELS:
        raise ValueError("Unsupported embedding provider")
    return preferred


def get_embedding_function(provider=None, credentials=None):
    from chromadb.utils import embedding_functions

    credentials = credentials if credentials is not None else Credentials.from_environment()
    provider = resolve_embedding_provider(provider, credentials)
    if provider == "openai":
        if not credentials.openai:
            raise ValueError("OpenAI embedding credentials required")
        function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=credentials.openai, model_name=DEFAULT_MODELS[provider]
        )
    else:
        function = embedding_functions.DefaultEmbeddingFunction()
    return function, provider, DEFAULT_MODELS[provider]
