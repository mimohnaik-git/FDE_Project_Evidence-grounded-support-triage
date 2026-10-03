"""Resolve existing provider options with explicitly injected credentials."""

from support_triage.runtime import Credentials, setting

DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001", "ollama": "llama3.1"}


def resolve_provider(preferred=None, credentials=None):
    credentials = credentials if credentials is not None else Credentials.from_environment()
    preferred = preferred or setting("LLM_PROVIDER", "auto")
    if preferred == "auto":
        return "openai" if credentials.openai else "anthropic" if credentials.anthropic else "ollama"
    if preferred not in DEFAULT_MODELS:
        raise ValueError("Unsupported chat provider")
    return preferred


def get_chat_model(provider=None, model=None, temperature=0.0, credentials=None):
    credentials = credentials if credentials is not None else Credentials.from_environment()
    provider = resolve_provider(provider, credentials)
    model = model or setting("LLM_MODEL") or DEFAULT_MODELS[provider]
    if provider == "openai":
        if not credentials.openai:
            raise ValueError("OpenAI credentials required")
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(model=model, temperature=temperature, api_key=credentials.openai)
    elif provider == "anthropic":
        if not credentials.anthropic:
            raise ValueError("Anthropic credentials required")
        from langchain_anthropic import ChatAnthropic

        chat = ChatAnthropic(model=model, temperature=temperature, api_key=credentials.anthropic)
    else:
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=model,
            temperature=temperature,
            base_url=setting("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    return chat, provider, model
