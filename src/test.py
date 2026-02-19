from __future__ import annotations

import os
from typing import List, Tuple


class LLMUnavailableError(RuntimeError):
    pass


def _get_ollama_client():
    try:
        import ollama
    except Exception as exc:
        raise LLMUnavailableError("ollama is not installed.") from exc

    host = os.environ.get("OLLAMA_HOST")
    api_key = os.environ.get("OLLAMA_API_KEY")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    if host or headers:
        return ollama.Client(host=host, headers=headers)
    return ollama.Client()


def _list_local_models(ollama_client) -> List[str]:
    try:
        payload = ollama_client.list()
    except Exception:
        return []

    models = payload.get("models", []) if isinstance(payload, dict) else []
    names = []
    for model in models:
        name = model.get("name") if isinstance(model, dict) else None
        if name:
            names.append(name)
    return names


def _resolve_model_name(ollama_client, requested: str) -> Tuple[str, List[str]]:
    available = _list_local_models(ollama_client)
    if not available:
        return requested, available

    if requested in available:
        return requested, available

    requested_base = requested.split(":", 1)[0]
    for name in available:
        if name.split(":", 1)[0] == requested_base:
            return name, available

    return requested, available


def _extract_generate_text(response) -> str:
    if response is None:
        return ""

    text = getattr(response, "response", None)
    if text:
        return text

    data = None
    if isinstance(response, dict):
        data = response
    elif hasattr(response, "model_dump"):
        data = response.model_dump()
    elif hasattr(response, "dict"):
        data = response.dict()

    if isinstance(data, dict):
        text = data.get("response")
        if text:
            return text

    return ""


def generate_answer() -> str:
    ollama_client = _get_ollama_client()

    prompt = "Say hello in one short sentence."
    requested_model = os.environ.get("OLLAMA_MODEL", "deepseek-coder:6.7b")
    model_name, available = _resolve_model_name(ollama_client, requested_model)

    try:
        response = ollama_client.generate(
            model=model_name,
            prompt=prompt,
            options={"temperature": 0},
        )
    except Exception as exc:
        message = str(exc).lower()
        if (
            "connect" in message
            or "connection" in message
            or "refused" in message
            or "failed to connect" in message
        ):
            raise LLMUnavailableError(
                "Ollama server not reachable. Ensure Ollama is running."
            ) from exc
        if "unauthorized" in message or "status code: 401" in message or "401" in message:
            raise LLMUnavailableError(
                "Ollama request unauthorized. Set OLLAMA_API_KEY for cloud models "
                "and OLLAMA_HOST if you are not using the local server."
            ) from exc
        if "not found" in message and "model" in message:
            hint = ""
            if available:
                preview = ", ".join(sorted(available)[:6])
                hint = f" Available models: {preview}."
            raise LLMUnavailableError(
                f"Model '{model_name}' not available locally.{hint} "
                f"Run `ollama pull {model_name}`."
            ) from exc
        raise

    return _extract_generate_text(response).strip()


print(generate_answer())
